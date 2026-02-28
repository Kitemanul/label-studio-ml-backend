"""
Tests for SAM 3 ML Backend.

Run with:
    pip install -r requirements-test.txt
    pytest test_api.py -v
"""

import pytest
import json
from model import SAM3MLBackend


_TEST_CONFIG = '''<View>
    <Image name="image" value="$image" zoom="true"/>
    <BrushLabels name="tag" toName="image">
        <Label value="Banana" background="#FF0000"/>
        <Label value="Orange" background="#0d14d3"/>
    </BrushLabels>
    <KeyPointLabels name="tag2" toName="image" smart="true">
        <Label value="Banana" background="#000000" showInline="true"/>
        <Label value="Orange" background="#000000" showInline="true"/>
    </KeyPointLabels>
    <RectangleLabels name="tag3" toName="image" smart="true">
        <Label value="Banana" background="#000000" showInline="true"/>
        <Label value="Orange" background="#000000" showInline="true"/>
    </RectangleLabels>
</View>'''

_TEST_TASK = [{
    'data': {
        'image': 'https://s3.amazonaws.com/htx-pub/datasets/images/125245483_152578129892066_7843809718842085333_n.jpg'
    }
}]


@pytest.fixture
def client():
    from _wsgi import init_app
    app = init_app(model_class=SAM3MLBackend)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health(client):
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'UP'


def test_predict_no_context_returns_text_prompted_results():
    """Without context, SAM3 should use label names as text prompts."""
    model = SAM3MLBackend(label_config=_TEST_CONFIG)
    result = model.predict(_TEST_TASK)
    # ModelResponse should contain predictions
    assert result is not None


def test_predict_with_keypoints():
    """With keypoint context, SAM3 should return brush masks."""
    model = SAM3MLBackend(label_config=_TEST_CONFIG)
    context = {
        'result': [{
            'original_width': 1080,
            'original_height': 1080,
            'image_rotation': 0,
            'value': {
                'x': 49.44,
                'y': 59.97,
                'width': 0.32,
                'labels': ['Banana'],
                'keypointlabels': ['Banana'],
            },
            'is_positive': True,
            'id': 'test123',
            'from_name': 'tag2',
            'to_name': 'image',
            'type': 'keypointlabels',
            'origin': 'manual',
        }]
    }
    result = model.predict(_TEST_TASK, context)
    assert result is not None


def test_predict_with_rectangle():
    """With rectangle context, SAM3 should return brush masks."""
    model = SAM3MLBackend(label_config=_TEST_CONFIG)
    context = {
        'result': [{
            'original_width': 1080,
            'original_height': 1080,
            'image_rotation': 0,
            'value': {
                'x': 20.0,
                'y': 30.0,
                'width': 40.0,
                'height': 50.0,
                'labels': ['Orange'],
                'rectanglelabels': ['Orange'],
            },
            'id': 'test456',
            'from_name': 'tag3',
            'to_name': 'image',
            'type': 'rectanglelabels',
            'origin': 'manual',
        }]
    }
    result = model.predict(_TEST_TASK, context)
    assert result is not None
