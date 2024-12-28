from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score    
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

class SaveReports:
    def __init__(self):
        self.results_dir = SaveReports.__create_results_directory()

    @staticmethod
    def __create_results_directory(base_folder="results"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        directory = os.path.join(base_folder, f"res-{timestamp}")
        os.makedirs(directory, exist_ok=True)
        return directory

    @staticmethod
    def __save_plot(fig, filename, save_dir):
        filepath = os.path.join(save_dir, filename)
        fig.savefig(filepath)
        plt.close(fig)

    # def save_confusion_matrix(self, true_classes, predicted_classes, target_names):
    #     conf_matrix = confusion_matrix(true_classes, predicted_classes)
    #     fig, ax = plt.subplots(figsize=(16, 16))
    #     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax)
    #     ax.set_title("Confusion Matrix")
    #     ax.set_xlabel("Predicted Classes")
    #     ax.set_ylabel("True Classes")
    #     SaveReports.__save_plot(fig, "confusion_matrix.png", self.results_dir)

    def save_confusion_matrix(self, true_classes, predicted_classes, target_names):
        conf_matrix = confusion_matrix(true_classes, predicted_classes)
        fig, ax = plt.subplots(figsize=(18, 18))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax, cbar=False)
        ax.set_title("Confusion Matrix", fontsize=20, pad=25)
        ax.set_xlabel("Predicted Classes", fontsize=18, labelpad=25)
        ax.set_ylabel("True Classes", fontsize=18, labelpad=25)
        ax.tick_params(axis='x', labelsize=16, rotation=45)
        ax.tick_params(axis='y', labelsize=16, rotation=0)
        plt.tight_layout()
        SaveReports.__save_plot(fig, "confusion_matrix.png", self.results_dir)


    # def save_class_distribution(self, true_classes, target_names):
    #     counts = np.bincount(true_classes)
    #     fig, ax = plt.subplots(figsize=(20, 15))
    #     ax.bar(target_names, counts, color='teal')
    #     ax.set_xlabel("Classes")
    #     ax.set_ylabel("Count")
    #     ax.set_title("Class Distribution")
    #     plt.xticks(rotation=90)
    #     SaveReports.__save_plot(fig, "class_distribution.png", self.results_dir)

    def save_class_distribution(self, true_classes, target_names):
        counts = np.bincount(true_classes)
        fig, ax = plt.subplots(figsize=(20, 15))
        ax.bar(target_names, counts, color='teal')
        ax.set_xlabel("Classes", fontsize=16, labelpad=30)
        ax.set_ylabel("Count", fontsize=16, labelpad=30)
        ax.set_title("Class Distribution", fontsize=18, pad=30)
        plt.xticks(rotation=90, fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        SaveReports.__save_plot(fig, "class_distribution.png", self.results_dir)


    def save_precision_recall_score(self, true_classes, predicted_classes, target_names):
        precision = precision_score(true_classes, predicted_classes, average=None)
        recall = recall_score(true_classes, predicted_classes, average=None)
        x = np.arange(len(target_names))
        width = 0.4

        fig, ax = plt.subplots(figsize=(20, 15))
        ax.bar(x - width/2, precision, width, label='Precision', color='skyblue')
        ax.bar(x + width/2, recall, width, label='Recall', color='orange')
        ax.set_xlabel("Classes")
        ax.set_ylabel("Scores")
        ax.set_title("Precision and Recall by Class")
        ax.set_xticks(x)
        ax.set_xticklabels(target_names, rotation=90)
        ax.legend()
        SaveReports.__save_plot(fig, "precision_recall.png", self.results_dir)

    def save_classification_report(self, true_classes, predicted_classes, target_names):
        class_report = classification_report(true_classes, predicted_classes, target_names=target_names)
        with open(os.path.join(self.results_dir, 'classification_report.txt'), 'w') as f:
            f.write(f"Classification Report:\n{class_report}")

if __name__ == "__main__":
    # Sample data to check if the code is working or not...
    true_classes = [0, 1, 0, 1, 2, 2]  # Example true classes
    predicted_classes = [0, 1, 1, 0, 2, 2]  # Example predicted classes
    target_names = ['Class 0', 'Class 1', 'Class 2']  # Example class names

    report_saver = SaveReports()
    report_saver.save_confusion_matrix(true_classes, predicted_classes, target_names)
    report_saver.save_class_distribution(true_classes, target_names)
    report_saver.save_precision_recall_score(true_classes, predicted_classes, target_names)
    report_saver.save_classification_report(true_classes, predicted_classes, target_names)
