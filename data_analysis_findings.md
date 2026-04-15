
# Data Analysis Findings

## Combined Data Preprocessing and GAN Augmentation Summary
### Q&A
Yes, the dataset has been successfully pre-processed and filtered. It now contains only the minority class images, which have been resized to 64x64 pixels and normalized to the range of [-1, 1]. This dataset is ready for use in a dataloader.

### Data Analysis Key Findings
*   The minority class in the `train_df` dataset was identified as label '1', with a total of 49 occurrences. The full label distribution was: '2' (2566), '3' (1781), '0' (724), and '1' (49).
*   A new DataFrame, `minority_class_df`, was successfully created, containing only the 49 images and their labels for the minority class '1'.
*   All 49 minority class images were successfully resized to 64x64 pixels and their pixel values were normalized to the range of [-1, 1].
*   During the process, a `SettingWithCopyWarning` was encountered and resolved by explicitly creating a copy of the DataFrame. A `DeprecationWarning` related to `mode='L'` in `Image.fromarray()` was also resolved by removing the deprecated parameter, allowing proper inference of the image mode.
*   Verification confirmed that the preprocessed images have the expected shape of (64, 64) and pixel values are correctly within the [-1, 1] range.

### Insights or Next Steps
*   The preprocessed minority class dataset is now in an optimal state for use in a PyTorch or TensorFlow `DataLoader` for tasks such as training a generative model (e.g., GANs or VAEs) focused on minority class samples.
*   Consider applying similar preprocessing steps consistently across all datasets (training, validation, test) to ensure uniformity if they are to be used with the same model architecture.

## CNN Model Training and Evaluation Results on Combined Dataset
### Q&A
The training and evaluation results of the CNN model on the combined dataset show an overall accuracy of 95.26% and an overall weighted F1-score of 95.28%. Performance varies slightly across classes, with Class 1 achieving a perfect 1.00 F1-score, while Class 0 has the lowest F1-score at 0.91.

### Data Analysis Key Findings
*   The CNN model was trained over 10 epochs. The training loss consistently decreased from approximately 0.8346 (Epoch 1) to 0.0521 (Epoch 10).
*   Validation accuracy improved significantly throughout training, starting at 63.73% (Epoch 1) and reaching a peak of 95.26% by Epoch 10.
*   The final evaluation on the validation set showed an overall accuracy of 0.9526 (95.26%) and an overall weighted F1-score of 0.9528 (95.28%).
*   Class-wise performance from the classification report indicates:
    *   **Class 1** demonstrated exceptional performance with a Precision, Recall, and F1-score of 1.00.
    *   **Class 2** and **Class 3** also performed very well, with F1-scores of 0.96 and 0.94, respectively.
    *   **Class 0** had slightly lower performance compared to others, with an F1-score of 0.91 (Precision: 0.88, Recall: 0.94).

### Insights or Next Steps
*   The model demonstrates strong classification capabilities for the given dataset, with high overall accuracy and F1-score.
*   Investigate the samples belonging to Class 0 to understand potential reasons for its slightly lower performance compared to other classes. This could involve reviewing image quality, feature commonality, or potential mislabeling within this specific class.

## Confusion Matrix Insights
### Q&A
The confusion matrix reveals strong classification performance overall. Class 1 (Minority Class) shows exceptional performance with 198 correct predictions out of 198 actual instances, indicating no misclassifications. Class 2 correctly predicted 505 instances, with minor misclassifications to Class 0 (9) and Class 3 (16). Similarly, Class 3 made 327 correct predictions, with 9 misclassified as Class 0 and 15 as Class 2. The main area for improvement is Class 0, where 136 instances were correctly identified, but 4 were misclassified as Class 2 and 5 as Class 3, which is relatively significant given its class size.

### Data Analysis Key Findings
*   A confusion matrix was successfully generated and displayed, visualizing the CNN model's classification performance.
*   The model exhibited strong performance for Class 1 (Minority Class), with 198 out of 198 instances correctly classified and no misclassifications.
*   For Class 2, the model achieved 505 correct predictions, with only 9 instances misclassified as Class 0 and 16 as Class 3.
*   Class 3 showed 327 correct predictions, with 9 instances misclassified as Class 0 and 15 as Class 2.
*   Class 0 was identified as the primary area for improvement, with 136 correct predictions but 4 misclassified as Class 2 and 5 as Class 3, suggesting some challenges for this class.

### Insights or Next Steps
*   The successful classification of Class 1 (Minority Class) with no errors highlights the effectiveness of the GAN-based data augmentation in balancing this class and improving the model's ability to learn its distinct features.
*   Further investigation into the misclassifications for Class 0 could involve analyzing the features of the misclassified instances to understand common patterns or similarities with other classes, potentially leading to targeted feature engineering or additional data augmentation for this specific class.
