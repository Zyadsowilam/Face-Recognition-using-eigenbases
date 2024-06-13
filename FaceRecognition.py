import numpy as np
import os
from PIL import Image

def face_recognition(test_image_path, q=35, threshold=0.7, var_thresholds=[85, 95]):
    def load_images_from_folder(folder_path):
        images = []
        labels = []
        for subdir in os.listdir(folder_path):
            subdir_path = os.path.join(folder_path, subdir)
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):
                    img_path = os.path.join(subdir_path, filename)
                    img = Image.open(img_path).convert('L')
                    images.append(np.array(img))
                    labels.append(int(subdir[1:]))
        return images, labels

    def compute_eigen(images):
        shape = images[0].shape
        Resize = np.resize(images, (len(images), shape[0] * shape[1]))
        mean = np.mean(Resize, axis=0)
        Resize_std = Resize - mean

        cov_mat = Resize_std @ Resize_std.T
        eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        eigenvector_converted = Resize_std.T @ eigenvectors
        eigen = eigenvector_converted.T / np.sqrt((eigenvector_converted.T ** 2).sum(axis=1, keepdims=True))
        return eigen, mean, shape, eigenvalues

    def explained_variance(eigenvalues, thresholds):
        total = sum(eigenvalues)
        explained_variance = [(i / total) * 100 for i in eigenvalues]
        explained_variance = np.round(explained_variance, 2)
        cum_explained_variance = np.cumsum(explained_variance)
        
        dimensions = []
        for threshold in thresholds:
            for dim in range(len(cum_explained_variance)):
                if cum_explained_variance[dim] >= threshold:
                    dimensions.append(dim + 1)
                    break
        return dimensions

    def project_images(images, eigen, mean, shape):
        projections = []
        for img in images:
            img_resized = np.array(Image.fromarray(img).resize(shape[::-1]))
            img_mean_centered = img_resized.reshape(-1) - mean
            projection = eigen.dot(img_mean_centered)
            projections.append(projection)
        return projections

    def predictSingle(img_index, testEF, q, threshold):
        highest_similarity = -1
        predicted_label = None
        
        E = testEF[img_index][:q]
        
        for i, train_img in enumerate(trainE):
            E_compare = train_img[:q]
            
            # Calculate cosine similarity
            similarity = np.dot(E, E_compare) / (np.linalg.norm(E) * np.linalg.norm(E_compare))
            
            # Check if similarity exceeds the threshold and is the highest so far
            if similarity > threshold and similarity > highest_similarity:
                highest_similarity = similarity
                predicted_label = train_labels[i]
        return predicted_label
        
    def predict(q, threshold):
        total = 0
        correct = 0
        TP, FP,  FN = 0, 0, 0
        
        for i in range(len(testE)):
            total += 1
           
            predicted_label = predictSingle(i, testE, q, threshold)
            
            if predicted_label == test_labels[i]:
                correct += 1
                if predicted_label is not None:
                    TP += 1  
            
            else:
                if predicted_label is not None:
                    FP += 1  
                else:
                    FN += 1  
        
        accuracy = correct / total
        false_positive_percentage = (FP / total) * 100
        false_negative_percentage = (FN / total) * 100
        true_positive_percentage = (TP / total) * 100
        
        
        return accuracy, false_positive_percentage, false_negative_percentage, true_positive_percentage

    # Load training images
    folder_path = r"D:\imgFR\archive"
    TestFolder = r"D:\imgFR\test"
    train_images, train_labels = load_images_from_folder(folder_path)
    test_images, test_labels =load_images_from_folder(TestFolder)

    eigen, mean, shape, eigenvalues = compute_eigen(train_images)
    dimensions = explained_variance(eigenvalues, var_thresholds)
    
    num_dimensions = dimensions[-1]  
    trainE = project_images(train_images, eigen[:num_dimensions], mean, shape)
    testE = project_images(test_images, eigen[:num_dimensions], mean, shape)

    accuracy, false_positive_percentage, false_negative_percentage, true_positive_percentage = predict(q, threshold)
    print("Classification Accuracy: ", accuracy)
    print("False Positive Percentage: ", false_positive_percentage)
    print("False Negative Percentage: ", false_negative_percentage)
    print("True Positive Percentage: ", true_positive_percentage)
    
    path_testing_list = []
    imgT = Image.open(test_image_path).convert('L')
    path_testing_list.append(np.array(imgT))
    project_selected = project_images(path_testing_list, eigen[:num_dimensions], mean, shape)
    
    predicted_label = predictSingle(0, project_selected, q, threshold)


    if predicted_label is not None:
        data_dir = r"D:\imgFR\archive"
        subject_dir = "s" + str(predicted_label)
        image_name = "2.pgm"
        predicted_image_path = os.path.join(data_dir, subject_dir, image_name)
    else:
        predicted_image_path = None

    return predicted_label, predicted_image_path
# test_image_path = r"D:\imgFR\test\s8\7.pgm"
# predicted_label = face_recognition(test_image_path, q=35, threshold=0.7)
# print("Predicted Label:", predicted_label)


# print("Subject Directory:", subject_dir)
# print("Image Name:", image_name)
