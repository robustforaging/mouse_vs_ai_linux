import os
import sys

def replace_nature_visual_encoder(encoder_file_path, custom_encoder_file_path):
    """
    Replaces the NatureVisualEncoder class definition in encoder.py with the
    contents of a custom encoder file.

    Args:
        encoder_file_path (str): Path to the encoder.py file.
        custom_encoder_file_path (str): Path to the file containing your custom
            encoder class definition.
    """
    try:
        with open(encoder_file_path, 'r') as f:
            encoder_code = f.read()
    except FileNotFoundError:
        print(f"Error: encoder.py not found at {encoder_file_path}")
        return

    try:
        with open(custom_encoder_file_path, 'r') as f:
            custom_encoder_code = f.read()
    except FileNotFoundError:
        print(f"Error: Custom encoder file not found at {custom_encoder_file_path}")
        return

    # Find the start and end of the NatureVisualEncoder class definition
    start_marker = "class NatureVisualEncoder(nn.Module):"
    # Find the next class definition or end of file
    lines = encoder_code.split('\n')
    start_index = encoder_code.find(start_marker)
    if start_index == -1:
        print("Error: Could not find NatureVisualEncoder definition in encoder.py")
        return

    # Find the next class definition
    next_class_index = encoder_code.find("\nclass ", start_index + len(start_marker))
    if next_class_index == -1:
        # If no next class, use the end of file
        next_class_index = len(encoder_code)

    # Extract the code before and after the NatureVisualEncoder definition
    code_before = encoder_code[:start_index]
    code_after = encoder_code[next_class_index:]

    # Construct the new encoder.py code
    new_encoder_code = code_before + custom_encoder_code + code_after

    # Write the modified code back to encoder.py
    try:
        with open(encoder_file_path, 'w') as f:
            f.write(new_encoder_code)
        print(f"encoder.py modified successfully.")
    except Exception as e:
        print(f"Error writing to encoder.py: {e}")
        return


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python replace_encoder.py <encoder_file_path> <custom_encoder_file_path>")
        sys.exit(1)

    encoder_file = sys.argv[1]
    custom_encoder_file = sys.argv[2]
    replace_nature_visual_encoder(encoder_file, custom_encoder_file)

