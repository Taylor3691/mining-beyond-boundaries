import matplotlib.pyplot as plt
def plot_deduplicate_comparison(initial_count, final_count):
    """
    Uses a bar chart to visualize the number of images before and after deduplication.
    """
    labels = ['Before Removal', 'After Removal']
    counts = [initial_count, final_count]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, counts, color=['#1f77b4', '#ff7f0e'])
    plt.ylabel('Number of Images')
    plt.title('Image Count Before and After Deduplication')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom')

    plt.show()