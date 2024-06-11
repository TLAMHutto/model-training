import matplotlib.pyplot as plt

# Function to plot images and predictions
def plot_images(images, predictions, labels):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i])
        plt.title(f'Pred: {predictions[i]}, Label: {labels[i]}')
        plt.axis('off')
    plt.show()

# Example usage
# Assuming `images`, `predictions`, and `labels` are available
plot_images(images, predictions, labels)
