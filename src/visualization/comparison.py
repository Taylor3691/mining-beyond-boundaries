import matplotlib.pyplot as plt
import cv2

def plot_deduplicate_comparison(initial_count, final_count):
    """
    Visualizes the number of images before and after the deduplication process using a bar chart.
    
    Args:
        initial_count (int): Original number of images in the dataset.
        final_count (int): Remaining number of images after processing.
    """
    labels = ['Before Removal', 'After Removal']
    counts = [initial_count, final_count]

    plt.figure(figsize=(8, 6))
    # Standard color scheme: Blue for Before, Orange for After
    bars = plt.bar(labels, counts, color=['#1f77b4', '#ff7f0e'])
    
    plt.ylabel('Number of Images')
    plt.title('Image Count Before and After Deduplication')
    
    # Display the exact count on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), 
                 va='bottom', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_phash_comparison(img1, img2, hash1, hash2, distance, title="pHash Comparison"):
    """
    Displays two images side-by-side with their Hash strings and Hamming distance for experimental analysis.
    
    Args:
        img1, img2 (numpy.ndarray): The two images to compare (BGR format from OpenCV).
        hash1, hash2 (str): Corresponding pHash strings.
        distance (int): Calculated Hamming distance between the two hashes.
        title (str): Title for the visualization.
    """
    # Convert BGR to RGB for correct display in Matplotlib
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Set title color: Green if similar (<=10), Red if different (>10)
    status_color = 'green' if distance <= 10 else 'red'
    fig.suptitle(f"{title}\nHamming Distance: {distance}", 
                 fontsize=14, fontweight='bold', color=status_color)
    
    # Plot Image 1 (Original/Reference)
    axes[0].imshow(img1_rgb)
    axes[0].set_title(f"Original / Side A\nHash: {hash1}", fontsize=9, family='monospace')
    axes[0].axis('off')
    
    # Plot Image 2 (Transformed/Duplicate)
    axes[1].imshow(img2_rgb)
    axes[1].set_title(f"Transformed / Side B\nHash: {hash2}", fontsize=9, family='monospace')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()