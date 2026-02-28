import logging
import os
import numpy as np
import torch
from typing import List, Dict, Optional
from uuid import uuid4
from PIL import Image

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.converter import brush

logger = logging.getLogger(__name__)

DEVICE = os.getenv('DEVICE', 'cuda')
SAM3_CHECKPOINT = os.getenv('SAM3_CHECKPOINT', None)

# Import sam3 components
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Build model at module level for efficiency
_build_kwargs = dict(enable_inst_interactivity=True)
if SAM3_CHECKPOINT:
    _build_kwargs['checkpoint_path'] = SAM3_CHECKPOINT
    logger.info(f'Loading SAM3 from local checkpoint: {SAM3_CHECKPOINT}')
else:
    logger.info('Loading SAM3 from HuggingFace (requires authentication)')

if DEVICE == 'cuda':
    torch.autocast(device_type='cuda', dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

sam3_model = build_sam3_image_model(**_build_kwargs)
processor = Sam3Processor(sam3_model)


class SAM3MLBackend(LabelStudioMLBase):
    """SAM 3 ML Backend for Label Studio.

    Supports three prompt modes:
    - Text prompts (pre-annotation): uses label names from BrushLabels
      as text prompts to segment all matching instances.
    - Point prompts (interactive): user clicks keypoints on the image.
    - Box prompts (interactive): user draws rectangles on the image.
    """

    def setup(self):
        self.set('model_version', f'{self.__class__.__name__}-v0.0.1')

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> ModelResponse:
        from_name, to_name, value = self.get_first_tag_occurence(
            'BrushLabels', 'Image'
        )

        if not context or not context.get('result'):
            # No interactive context — run text-prompted pre-annotation
            return self._predict_text(tasks, from_name, to_name, value)

        # Interactive mode: parse keypoints and rectangles from context
        return self._predict_interactive(tasks, context, from_name, to_name, value)

    # ------------------------------------------------------------------
    # Text-prompted pre-annotation
    # ------------------------------------------------------------------
    def _predict_text(
        self, tasks: List[Dict], from_name: str, to_name: str, value: str
    ) -> ModelResponse:
        """Use label names as text prompts to segment all matching instances."""
        labels_attrs = self.label_interface.get_control(from_name).labels_attrs
        if not labels_attrs:
            return ModelResponse(predictions=[])

        label_names = list(labels_attrs.keys())
        all_predictions = []

        for task in tasks:
            img_path = task['data'][value]
            image = self._load_image(img_path, task)
            inference_state = processor.set_image(image)

            results = []
            total_score = 0.0
            count = 0

            for label_name in label_names:
                try:
                    output = processor.set_text_prompt(
                        state=inference_state, prompt=label_name
                    )
                except Exception as e:
                    logger.warning(
                        f'Text prompt failed for "{label_name}": {e}'
                    )
                    continue

                masks = output['masks']
                scores = output['scores']

                if isinstance(masks, torch.Tensor):
                    masks = masks.cpu().numpy()
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()

                image_height, image_width = masks.shape[-2:]

                for i in range(len(scores)):
                    mask = masks[i].astype(np.uint8)
                    score = float(scores[i])
                    label_id = str(uuid4())[:4]
                    rle = brush.mask2rle(mask * 255)
                    results.append({
                        'id': label_id,
                        'from_name': from_name,
                        'to_name': to_name,
                        'original_width': image_width,
                        'original_height': image_height,
                        'image_rotation': 0,
                        'value': {
                            'format': 'rle',
                            'rle': rle,
                            'brushlabels': [label_name],
                        },
                        'score': score,
                        'type': 'brushlabels',
                        'readonly': False,
                    })
                    total_score += score
                    count += 1

            all_predictions.append({
                'result': results,
                'model_version': self.get('model_version'),
                'score': total_score / max(count, 1),
            })

        return ModelResponse(predictions=all_predictions)

    # ------------------------------------------------------------------
    # Interactive segmentation (point / box prompts)
    # ------------------------------------------------------------------
    def _predict_interactive(
        self,
        tasks: List[Dict],
        context: Dict,
        from_name: str,
        to_name: str,
        value: str,
    ) -> ModelResponse:
        image_width = context['result'][0]['original_width']
        image_height = context['result'][0]['original_height']

        point_coords = []
        point_labels = []
        input_box = None
        selected_label = None

        for ctx in context['result']:
            x = ctx['value']['x'] * image_width / 100
            y = ctx['value']['y'] * image_height / 100
            ctx_type = ctx['type']
            selected_label = ctx['value'][ctx_type][0]

            if ctx_type == 'keypointlabels':
                point_labels.append(int(ctx.get('is_positive', 0)))
                point_coords.append([int(x), int(y)])
            elif ctx_type == 'rectanglelabels':
                box_width = ctx['value']['width'] * image_width / 100
                box_height = ctx['value']['height'] * image_height / 100
                input_box = [int(x), int(y), int(box_width + x), int(box_height + y)]

        logger.debug(
            f'Point coords: {point_coords}, point labels: {point_labels}, '
            f'input box: {input_box}'
        )

        img_path = tasks[0]['data'][value]
        image = self._load_image(img_path, tasks[0])
        inference_state = processor.set_image(image)

        np_point_coords = (
            np.array(point_coords, dtype=np.float32) if point_coords else None
        )
        np_point_labels = (
            np.array(point_labels, dtype=np.float32) if point_labels else None
        )
        np_box = (
            np.array(input_box, dtype=np.float32)[None, :] if input_box else None
        )

        masks, scores, logits = sam3_model.predict_inst(
            inference_state,
            point_coords=np_point_coords,
            point_labels=np_point_labels,
            box=np_box,
            multimask_output=True,
        )

        # Pick the best mask
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        mask = masks[0, :, :].astype(np.uint8)
        prob = float(scores[0])

        predictions = self._format_brush_results(
            masks=[mask],
            probs=[prob],
            width=image_width,
            height=image_height,
            from_name=from_name,
            to_name=to_name,
            label=selected_label,
        )

        return ModelResponse(predictions=predictions)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_image(self, url: str, task: Dict) -> Image.Image:
        """Download/resolve the image URL and return a PIL Image."""
        local_path = self.get_local_path(url, task_id=task.get('id'))
        return Image.open(local_path).convert('RGB')

    @staticmethod
    def _format_brush_results(
        masks, probs, width, height, from_name, to_name, label
    ) -> List[Dict]:
        results = []
        total_prob = 0.0
        for mask, prob in zip(masks, probs):
            label_id = str(uuid4())[:4]
            rle = brush.mask2rle(mask * 255)
            total_prob += prob
            results.append({
                'id': label_id,
                'from_name': from_name,
                'to_name': to_name,
                'original_width': width,
                'original_height': height,
                'image_rotation': 0,
                'value': {
                    'format': 'rle',
                    'rle': rle,
                    'brushlabels': [label],
                },
                'score': prob,
                'type': 'brushlabels',
                'readonly': False,
            })

        return [{
            'result': results,
            'score': total_prob / max(len(results), 1),
        }]
