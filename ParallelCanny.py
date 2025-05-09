from mpi4py import MPI
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os

def parallel_canny_approach_a(image_path, low_threshold=100, high_threshold=200):
   
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Print debug info from each process
    print(f"[Process {rank}] Starting. Current directory: {os.getcwd()}")
    
    # Check if file exists
    if rank == 0:
        if os.path.exists(image_path):
            print(f"[Process {rank}] Found image file: {image_path}")
        else:
            print(f"[Process {rank}] ERROR: Image file not found: {image_path}")
            comm.Abort(1)
    
    # All processes read the image
    print(f"[Process {rank}] Attempting to read image...")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"[Process {rank}] ERROR: Could not read image at {image_path}")
        if rank == 0:
            comm.Abort(1)
    
    start_time = time.time()
    print(f"[Process {rank}] Successfully read image of shape {image.shape}")
    
    # Divide work among processes
    height, width = image.shape
    rows_per_process = height // size + (1 if rank < height % size else 0)
    start_row = sum(height // size + (1 if i < height % size else 0) for i in range(rank))
    
    print(f"[Process {rank}] Assigned rows {start_row} to {start_row + rows_per_process} out of {height}")
    
    # Each process computes Canny on its portion
    # Add a small overlap to handle boundary pixels
    overlap = 2  # 2 pixels overlap on each side
    
    # Calculate the extended region with overlap
    local_start = max(0, start_row - overlap)
    local_end = min(height, start_row + rows_per_process + overlap)
    local_image = image[local_start:local_end, :]
    
    print(f"[Process {rank}] Processing local image section of shape {local_image.shape}")
    
    # Apply Canny edge detection to local portion
    print(f"[Process {rank}] Running Canny edge detection...")
    local_edges = cv2.Canny(local_image, low_threshold, high_threshold)
    print(f"[Process {rank}] Canny edge detection complete")
    
    # Remove overlap
    if rank > 0:
        start_offset = overlap
    else:
        start_offset = 0
        
    if rank < size - 1:
        end_offset = -overlap if local_end < height else None
    else:
        end_offset = None
    
    # Extract the actual portion without overlaps
    if end_offset is not None:
        local_edges = local_edges[start_offset:end_offset, :]
    else:
        local_edges = local_edges[start_offset:, :]
    
    if rank == 0:
        print(f"Local processing completed in {time.time() - start_time:.2f} seconds")
    
    print(f"[Process {rank}] Local processing completed, preparing to gather results")
    
    # Gather results
    gather_start = time.time()
    
    # Create a list to store all edge portions
    print(f"[Process {rank}] Gathering edge portions...")
    all_edges = comm.gather(local_edges, root=0)
    print(f"[Process {rank}] Gather operation completed")
    
    if rank == 0:
        print(f"[Process {rank}] Gathering completed in {time.time() - gather_start:.2f} seconds")
        
        # Combine all edge portions
        print(f"[Process {rank}] Combining edge portions...")
        combined_edges = np.vstack(all_edges)
        print(f"[Process {rank}] Final edge map shape: {combined_edges.shape}")
        
        return combined_edges
    return None

def main():
    """Main function to run parallel Canny edge detection"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    print(f"[Process {rank}/{size}] Process started")
    
    image_path = 'image.jpg'  # Replace with your image path
    
    if rank == 0:
        print(f"=== Starting parallel Canny edge detection with {size} processes ===")
        print(f"[Process {rank}] Looking for image at: {os.path.abspath(image_path)}")
        total_start = time.time()
    
    # Print barrier to ensure all processes have started
    comm.Barrier()
    print(f"[Process {rank}/{size}] Past initial barrier, beginning edge detection")
    
    # Run simplified parallel Canny edge detection (Approach A)
    edges = parallel_canny_approach_a(image_path, low_threshold=50, high_threshold=150)
    
    # Only rank 0 displays the result
    if rank == 0:
        print(f"Total processing time: {time.time() - total_start:.2f} seconds")
        
        # Display results
        original = cv2.imread(image_path)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(121), plt.imshow(original_rgb)
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(edges, cmap='gray')
        plt.title('Edge Image (Parallel)'), plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.show()
        
        # Compare with OpenCV implementation
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        cv_start = time.time()
        cv_edges = cv2.Canny(img, 50, 150)
        cv_time = time.time() - cv_start
        
        print(f"OpenCV sequential implementation time: {cv_time:.2f} seconds")
        
        plt.figure(figsize=(12, 6))
        plt.subplot(121), plt.imshow(edges, cmap='gray')
        plt.title(f'Our Parallel Implementation ({time.time() - total_start:.2f}s)'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(cv_edges, cmap='gray')
        plt.title(f'OpenCV Implementation ({cv_time:.2f}s)'), plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.show()
        
        # Save results
        cv2.imwrite('parallel_edges.jpg', edges)
        cv2.imwrite('opencv_edges.jpg', cv_edges)

if __name__ == "__main__":
    main()