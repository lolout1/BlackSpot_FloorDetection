from PIL import Image
import os

def convert_webp_to_jpg_same_name(input_directory):
    """
    Convert WebP files to JPG format while keeping the original filenames.
    The script converts the files in place, maintaining the original filename
    but changing the extension from .webp to .jpg.
    
    Args:
        input_directory (str): Directory containing WebP files
    """
    # Track our conversion statistics for reporting
    files_processed = 0
    files_skipped = 0
    
    # Process each file in the input directory
    for filename in os.listdir(input_directory):
        # Check if the file is a WebP image
        if filename.lower().endswith('.webp'):
            try:
                # Create full path for input file
                input_path = os.path.join(input_directory, filename)
                
                # Create output path with same name but .jpg extension
                # We use splitext to separate the name and extension
                base_name = os.path.splitext(filename)[0]
                output_filename = base_name + '.jpg'
                output_path = os.path.join(input_directory, output_filename)
                
                # Open and convert the image
                with Image.open(input_path) as img:
                    # Handle images with transparency by converting to RGB
                    if img.mode in ('RGBA', 'LA'):
                        # Create a white background
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        # Paste the image on the background
                        background.paste(img, mask=img.split()[-1])
                        # Save the image with the same name
                        background.save(output_path, 'JPEG', quality=95)
                    else:
                        # Save directly if already in RGB mode
                        img.save(output_path, 'JPEG', quality=95)
                
                # Remove the original WebP file after successful conversion
                os.remove(input_path)
                
                print(f"Successfully converted '{filename}' to '{output_filename}'")
                files_processed += 1
                
            except Exception as e:
                print(f"Error converting '{filename}': {str(e)}")
                files_skipped += 1
    
    # Print summary of the conversion process
    print(f"\nConversion complete!")
    print(f"Successfully converted: {files_processed} files")
    if files_skipped > 0:
        print(f"Skipped: {files_skipped} files due to errors")

if __name__ == "__main__":
    # Define your input directory where the WebP files are located
    input_dir = "trainMulti"    # Change this to your directory containing WebP files
    
    # Run the conversion process
    convert_webp_to_jpg_same_name(input_dir)
