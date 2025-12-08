import os
import cv2
import pytest
import numpy as np
from face_detect import detect_faces

# --- Fixtures ---
# Fixtures are setup steps that run before the tests.
# This one creates a temporary black image so we don't depend on your upload.
@pytest.fixture
def setup_dummy_image():
    # Create a simple 100x100 black image
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    filename = 'test_input_dummy.jpg'
    cv2.imwrite(filename, dummy_img)
    
    # Pass the filename to the test function
    yield filename
    
    # Cleanup: Delete the file after the test finishes
    if os.path.exists(filename):
        os.remove(filename)
    if os.path.exists('test_output.jpg'):
        os.remove('test_output.jpg')

# --- Tests ---

def test_detection_on_dummy_image(setup_dummy_image):
    """
    Test 1: Does the code run without crashing on a valid image?
    Expected: It should return an integer (number of faces, likely 0 for a black square).
    """
    input_file = setup_dummy_image
    
    # We call your function using the arguments we added in Step 2
    result = detect_faces(input_path=input_file, output_path='test_output.jpg')
    
    # Check if result is a number
    assert isinstance(result, int)
    # Check if an output file was actually created
    assert os.path.exists('test_output.jpg')

def test_missing_file_handled():
    """
    Test 2: Does the code handle missing files gracefully?
    Expected: It should print an error and return 0 (not crash).
    """
    # We deliberately ask for a file that doesn't exist
    result = detect_faces(input_path='non_existent_ghost_file.jpg')
    
    assert result == 0

def test_real_image_exists():
    """
    Test 3: Checks if your real 'input.jpg' is actually in the repo.
    This is just a sanity check for your project.
    """
    assert os.path.exists('input.jpg') == True
