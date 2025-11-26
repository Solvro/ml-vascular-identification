# Summary of Experimental Results

This directory contains the aggregated results from the Open Set Recognition experiments, formatted as CSV tables for the paper.

## 📊 Table Descriptions

### **Table 2: Architecture Comparison**
*   **Filename:** `table2_architecture_comparison.csv`
*   **Description:** Compares different backbone architectures (ResNet50-CBAM, Attention U-Net, Simple CNN, U-Net Embedding) on the MMCBNU dataset.
*   **Key Metrics:** OSCR, AUROC, EER, Rank-1 Accuracy, Inference Time.
*   **Goal:** To demonstrate that the ResNet50-CBAM architecture provides the best trade-off between performance and complexity.

### **Table 3: Generalization Across Datasets**
*   **Filename:** `table3_generalization.csv`
*   **Description:** Evaluates the best performing model (ResNet50-CBAM) across four different datasets: MMCBNU, FYO, Dorsal, and UTFVP.
*   **Key Metrics:** OSCR, AUROC, EER, Rank-1 Accuracy.
*   **Goal:** To assess the robustness and generalization capability of the proposed method on datasets with varying characteristics (image quality, number of subjects).

### **Table 4a: Ablation Study - Loss Functions**
*   **Filename:** `table4a_ablation_loss.csv`
*   **Description:** Analyzes the impact of different loss functions (Triplet, Triplet+Center, Contrastive) on the model's performance.
*   **Fixed Parameters:** Dataset=MMCBNU, Model=ResNet50-CBAM, Dim=256.
*   **Goal:** To justify the choice of the Triplet+Center loss for the final framework.

### **Table 4b: Ablation Study - Embedding Dimension**
*   **Filename:** `table4b_ablation_dim.csv`
*   **Description:** Investigates how the size of the feature embedding vector (128, 256, 384, 512) affects recognition accuracy.
*   **Fixed Parameters:** Dataset=MMCBNU, Model=ResNet50-CBAM, Loss=Triplet+Center.
*   **Goal:** To find the optimal embedding size that balances discriminative power and computational efficiency.

### **Table 4c: Impact of K-Nearest Neighbors**
*   **Filename:** `table4c_knn.csv`
*   **Description:** Shows the performance of the Open Set Recognition system when using different values of $K$ (1, 3, 5) for the K-NN classifier during inference.
*   **Goal:** To analyze the sensitivity of the system to the number of neighbors used for decision making.
