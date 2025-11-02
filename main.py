import sys
import os
import csv
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QFormLayout, QCheckBox, QSpinBox, QDoubleSpinBox,
    QMessageBox, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QTextEdit, QGroupBox
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
from preprocessing import preprocess_image, visualize_contextual_maps
from features import extract_features, get_chain_code_details, visualize_chain_code
from selection import PCAProjector  # new
from classifier import KerasClassifier  # new

def image_to_qimage(img: np.ndarray) -> QImage:
    if img is None:
        return QImage()
    if img.ndim == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        return qimg.copy()
    if img.ndim == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return qimg.copy()
    return QImage()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fingerprint Liveness - Preprocessing & Features")
        self.resize(1100, 700)

        self.current_image = None  # original (grayscale)
        self.processed_image = None  # result after preprocessing
        self.current_path = None

        self.dataset = []  # list of {path, features, label}
        self.feature_names = None  # order used for PCA
        self.pca = None  # PCAProjector
        self.clf = None  # KerasClassifier
        self.last_prediction = None  # (label, prob)

        tabs = QTabWidget()
        tabs.addTab(self._build_preprocess_tab(), "Preprocessing")
        tabs.addTab(self._build_features_tab(), "Feature Extraction")
        tabs.addTab(self._build_selection_tab(), "Feature Selection")  # new
        tabs.addTab(self._build_classify_tab(), "Classification")      # new
        tabs.addTab(self._build_results_tab(), "Results")              # new
        self.setCentralWidget(tabs)

    def _build_preprocess_tab(self) -> QWidget:
        page = QWidget()
        root = QHBoxLayout(page)

        # Left controls
        controls = QVBoxLayout()
        btn_row = QHBoxLayout()
        self.btn_load = QPushButton("Load Image")
        self.btn_save = QPushButton("Save Processed")
        self.btn_save.setEnabled(False)
        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_save)
        controls.addLayout(btn_row)

        form = QFormLayout()
        self.lbl_info = QLabel("No image")

        # Contextual filtering params
        self.chk_apply_ctx = QCheckBox("Apply Contextual Filtering")
        self.chk_apply_ctx.setChecked(True)
        self.spn_block = QSpinBox(); self.spn_block.setRange(8, 64); self.spn_block.setValue(16)
        self.dsp_min_wl = QDoubleSpinBox(); self.dsp_min_wl.setRange(2.0, 32.0); self.dsp_min_wl.setSingleStep(0.5); self.dsp_min_wl.setValue(2.0)
        self.dsp_max_wl = QDoubleSpinBox(); self.dsp_max_wl.setRange(2.0, 64.0); self.dsp_max_wl.setSingleStep(0.5); self.dsp_max_wl.setValue(2.0)
        self.dsp_gamma = QDoubleSpinBox(); self.dsp_gamma.setRange(0.2, 1.5); self.dsp_gamma.setSingleStep(0.1); self.dsp_gamma.setValue(0.5)
        self.dsp_gain = QDoubleSpinBox(); self.dsp_gain.setRange(0.1, 3.0); self.dsp_gain.setSingleStep(0.1); self.dsp_gain.setValue(0.9)

        form.addRow("Info:", self.lbl_info)
        form.addRow(self.chk_apply_ctx)
        form.addRow("Block size (px):", self.spn_block)
        form.addRow("Min wavelength (px):", self.dsp_min_wl)
        form.addRow("Max wavelength (px):", self.dsp_max_wl)
        form.addRow("Gamma (aspect):", self.dsp_gamma)
        form.addRow("Gain:", self.dsp_gain)

        controls.addLayout(form)

        act_row = QHBoxLayout()
        self.btn_apply = QPushButton("Apply")
        self.btn_reset = QPushButton("Reset")
        self.btn_apply.setEnabled(False)
        self.btn_reset.setEnabled(False)
        act_row.addWidget(self.btn_apply)
        act_row.addWidget(self.btn_reset)
        controls.addLayout(act_row)
        controls.addStretch(1)

        # Right images
        views = QHBoxLayout()
        self.lbl_orig = QLabel("Original")
        self.lbl_proc = QLabel("Processed")
        for lab in (self.lbl_orig, self.lbl_proc):
            lab.setAlignment(Qt.AlignCenter)
            lab.setMinimumSize(400, 400)
            lab.setStyleSheet("QLabel { background: #222; color: #ccc; border: 1px solid #444; }")
        views.addWidget(self._wrap_with_title(self.lbl_orig, "Original"))
        views.addWidget(self._wrap_with_title(self.lbl_proc, "Processed"))

        root.addLayout(controls, 0)
        root.addLayout(views, 1)

        # Signals
        self.btn_load.clicked.connect(self.on_load)
        self.btn_apply.clicked.connect(self.on_apply)
        self.btn_reset.clicked.connect(self.on_reset)
        self.btn_save.clicked.connect(self.on_save_processed)

        return page

    def _build_features_tab(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)

        top = QHBoxLayout()
        self.chk_use_processed = QCheckBox("Use preprocessed image if available")
        self.chk_use_processed.setChecked(True)
        top.addWidget(self.chk_use_processed)
        top.addStretch(1)

        # Chain Code options
        grp = QGroupBox("Chain Code options")
        gl = QFormLayout(grp)
        self.chk_cc_otsu = QCheckBox("Binarize with Otsu")
        self.chk_cc_otsu.setChecked(True)
        self.spn_cc_thresh = QSpinBox(); self.spn_cc_thresh.setRange(1, 255); self.spn_cc_thresh.setValue(128); self.spn_cc_thresh.setEnabled(False)
        self.chk_cc_8conn = QCheckBox("8-connectivity")
        self.chk_cc_8conn.setChecked(True)
        gl.addRow(self.chk_cc_otsu)
        gl.addRow("Manual threshold:", self.spn_cc_thresh)
        gl.addRow(self.chk_cc_8conn)

        # actions
        act = QHBoxLayout()
        self.btn_compute = QPushButton("Compute Features")
        self.btn_export = QPushButton("Export CSV")
        self.btn_export.setEnabled(False)
        # dataset add (kept)
        self.cmb_label = QComboBox(); self.cmb_label.addItems(["Real", "Fake"])
        self.btn_add_dataset = QPushButton("Add to dataset"); self.btn_add_dataset.setEnabled(False)
        act.addWidget(self.btn_compute); act.addWidget(self.btn_export); act.addStretch(1)
        act.addWidget(QLabel("Label:")); act.addWidget(self.cmb_label); act.addWidget(self.btn_add_dataset)

        self.tbl = QTableWidget(0, 2)
        self.tbl.setHorizontalHeaderLabels(["Feature", "Value"])
        self.tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)

        root.addLayout(top)
        root.addWidget(grp)
        root.addLayout(act)
        root.addWidget(self.tbl, 1)

        # signals
        self.chk_cc_otsu.toggled.connect(lambda v: self.spn_cc_thresh.setEnabled(not v))
        self.btn_compute.clicked.connect(self.on_compute_features)
        self.btn_export.clicked.connect(self.on_export_csv)
        self.btn_add_dataset.clicked.connect(self.on_add_to_dataset)

        return page

    def _build_selection_tab(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)

        top = QHBoxLayout()
        self.lbl_ds_count = QLabel("Dataset: 0 samples")
        self.spn_pca_k = QSpinBox(); self.spn_pca_k.setRange(2, 128); self.spn_pca_k.setValue(10)
        self.btn_fit_pca = QPushButton("Fit PCA")
        self.btn_apply_pca = QPushButton("Transform last features")
        self.btn_apply_pca.setEnabled(False)
        top.addWidget(self.lbl_ds_count); top.addStretch(1)
        top.addWidget(QLabel("n_components:")); top.addWidget(self.spn_pca_k)
        top.addWidget(self.btn_fit_pca); top.addWidget(self.btn_apply_pca)

        self.txt_pca = QTextEdit(); self.txt_pca.setReadOnly(True)

        root.addLayout(top)
        root.addWidget(self.txt_pca, 1)

        self.btn_fit_pca.clicked.connect(self.on_fit_pca)
        self.btn_apply_pca.clicked.connect(self.on_apply_pca)
        return page

    def _build_classify_tab(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)

        row = QHBoxLayout()
        self.btn_load_model = QPushButton("Load Keras model")
        self.lbl_model = QLabel("No model")
        row.addWidget(self.btn_load_model); row.addWidget(self.lbl_model); row.addStretch(1)

        form = QFormLayout()
        self.spn_in_w = QSpinBox(); self.spn_in_h = QSpinBox()
        for s in (self.spn_in_w, self.spn_in_h):
            s.setRange(32, 1024)
        self.spn_in_w.setValue(224); self.spn_in_h.setValue(224)
        self.chk_gray_in = QCheckBox("Grayscale input")
        self.chk_gray_in.setChecked(True)
        self.chk_norm01 = QCheckBox("Normalize [0,1]")
        self.chk_norm01.setChecked(True)
        self.spn_thresh = QDoubleSpinBox(); self.spn_thresh.setRange(0.0, 1.0); self.spn_thresh.setSingleStep(0.05); self.spn_thresh.setValue(0.5)
        form.addRow("Input W:", self.spn_in_w)
        form.addRow("Input H:", self.spn_in_h)
        form.addRow(self.chk_gray_in)
        form.addRow(self.chk_norm01)
        form.addRow("Threshold:", self.spn_thresh)

        act = QHBoxLayout()
        self.btn_predict = QPushButton("Predict current image")
        self.btn_predict.setEnabled(False)
        self.lbl_pred = QLabel("Prediction: -")
        act.addWidget(self.btn_predict); act.addStretch(1); act.addWidget(self.lbl_pred)

        root.addLayout(row)
        root.addLayout(form)
        root.addLayout(act)
        root.addStretch(1)

        self.btn_load_model.clicked.connect(self.on_load_model)
        self.btn_predict.clicked.connect(self.on_predict)
        return page

    def _build_results_tab(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        self.txt_results = QTextEdit(); self.txt_results.setReadOnly(True)
        root.addWidget(self.txt_results, 1)
        return page

    def _wrap_with_title(self, widget: QWidget, title: str) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lab = QLabel(title)
        lab.setAlignment(Qt.AlignCenter)
        lay.addWidget(lab)
        lay.addWidget(widget, 1)
        return w

    def on_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open fingerprint", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            QMessageBox.warning(self, "Error", "Failed to load image.")
            return
        self.current_image = img
        self.processed_image = None
        self.current_path = path
        self.lbl_info.setText(f"{os.path.basename(path)} | {img.shape[1]}x{img.shape[0]}")

        self._show_image(self.lbl_orig, img)
        self._show_image(self.lbl_proc, None)

        self.btn_apply.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.btn_compute.setEnabled(True)
        self.btn_export.setEnabled(False)
        self.btn_add_dataset.setEnabled(False)
        self._log("Loaded image: " + os.path.basename(path))

    def _show_image(self, label: QLabel, img: np.ndarray | None):
        if img is None:
            label.setPixmap(QPixmap())
            label.setText("No image")
            return
        qimg = image_to_qimage(img)
        pm = QPixmap.fromImage(qimg)
        pm = pm.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pm)
        label.setText("")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep previews properly scaled
        self._show_image(self.lbl_orig, self.current_image)
        self._show_image(self.lbl_proc, self.processed_image)

    def on_apply(self):
        if self.current_image is None:
            return
        opts = {
            "apply": self.chk_apply_ctx.isChecked(),
            "block": int(self.spn_block.value()),
            "min_wl": float(self.dsp_min_wl.value()),
            "max_wl": float(self.dsp_max_wl.value()),
            "gamma": float(self.dsp_gamma.value()),
            "gain": float(self.dsp_gain.value()),
        }
        # get processed image and diagnostic maps
        proc, energy, freq, theta = preprocess_image(self.current_image, return_maps=True, **opts)
        self.processed_image = proc
        self._show_image(self.lbl_proc, proc)
        self.btn_save.setEnabled(True)
        # show separate figure with (b) energy, (c) frequency, (d) orientation
        try:
            visualize_contextual_maps(energy, freq, theta)
        except Exception as e:
            QMessageBox.information(self, "Visualization", f"Could not show contextual maps:\n{e}")

    def on_reset(self):
        if self.current_image is None:
            return
        self.processed_image = None
        self._show_image(self.lbl_proc, None)
        self.chk_apply_ctx.setChecked(True)
        self.spn_block.setValue(16)
        self.dsp_min_wl.setValue(4.0)
        self.dsp_max_wl.setValue(16.0)
        self.dsp_gamma.setValue(0.5)
        self.dsp_gain.setValue(1.0)
        self.btn_save.setEnabled(False)

    def on_save_processed(self):
        if self.processed_image is None:
            return
        base = "processed_" + (os.path.basename(self.current_path) if self.current_path else "image.png")
        path, _ = QFileDialog.getSaveFileName(self, "Save processed image", base, "PNG (*.png);;JPG (*.jpg)")
        if not path:
            return
        ok = cv2.imwrite(path, self.processed_image)
        if not ok:
            QMessageBox.warning(self, "Error", "Failed to save image.")

    def on_compute_features(self):
        img = self.processed_image if (self.chk_use_processed.isChecked() and self.processed_image is not None) else self.current_image
        if img is None:
            QMessageBox.information(self, "Info", "Load an image first.")
            return
        feat_opts = dict(
            use_otsu=self.chk_cc_otsu.isChecked(),
            manual_thresh=int(self.spn_cc_thresh.value()),
            eight_conn=self.chk_cc_8conn.isChecked(),
        )
        feats = extract_features(img, **feat_opts)
        self._populate_table(feats)
        self.btn_export.setEnabled(True)
        self.btn_add_dataset.setEnabled(True)
        self.last_features = feats
        # Chain-code diagnostic visualization (separate figure)
        try:
            details = get_chain_code_details(img,
                                             use_otsu=self.chk_cc_otsu.isChecked(),
                                             manual_thresh=int(self.spn_cc_thresh.value()),
                                             eight_conn=self.chk_cc_8conn.isChecked())
            visualize_chain_code(details)
        except Exception as e:
            QMessageBox.information(self, "Visualization", f"Could not show chain code figure:\n{e}")

    def on_add_to_dataset(self):
        if not hasattr(self, "last_features"):
            return
        label = self.cmb_label.currentText()
        self.dataset.append({
            "path": self.current_path or "memory",
            "features": dict(self.last_features),
            "label": label
        })
        self.lbl_ds_count.setText(f"Dataset: {len(self.dataset)} samples")
        self._log(f"Added sample to dataset with label={label}.")

    def on_export_csv(self):
        if self.tbl.rowCount() == 0:
            return
        base = "features.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", base, "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["feature", "value"])
            for r in range(self.tbl.rowCount()):
                name = self.tbl.item(r, 0).text()
                val = self.tbl.item(r, 1).text()
                writer.writerow([name, val])

    def on_fit_pca(self):
        if len(self.dataset) < 2:
            QMessageBox.information(self, "Info", "Need at least 2 samples in dataset.")
            return
        # build consistent feature matrix
        names = sorted({k for s in self.dataset for k in s["features"].keys()})
        X = np.array([[s["features"].get(n, 0.0) for n in names] for s in self.dataset], dtype=np.float32)
        k = min(int(self.spn_pca_k.value()), X.shape[1], X.shape[0])
        self.pca = PCAProjector(n_components=k).fit(X)
        self.feature_names = names
        evr = self.pca.explained_variance_ratio_
        txt = []
        txt.append(f"PCA fitted: k={k}, D={X.shape[1]}, N={X.shape[0]}")
        txt.append("Explained variance ratio (first 10): " + ", ".join([f"{v:.3f}" for v in evr[:10]]))
        self.txt_pca.setPlainText("\n".join(txt))
        self.btn_apply_pca.setEnabled(True)
        self._log("PCA fitted.")

    def on_apply_pca(self):
        if self.pca is None or not hasattr(self, "last_features"):
            return
        x = np.array([[self.last_features.get(n, 0.0) for n in self.feature_names]], dtype=np.float32)
        z = self.pca.transform(x)[0]
        self._log("PCA(0:5): " + ", ".join([f"{v:.3f}" for v in z[:5]]))
        self.txt_pca.append("Last sample PCA: " + ", ".join([f"{v:.3f}" for v in z]))

    def on_load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Keras model", "", "Keras (*.h5);;SavedModel (saved_model.pb)")
        if not path:
            return
        try:
            self.clf = KerasClassifier(path)
            self.lbl_model.setText(os.path.basename(path))
            self.btn_predict.setEnabled(True)
            self._log("Loaded model.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load model:\n{e}")

    def on_predict(self):
        if self.clf is None:
            return
        img = self.processed_image if (self.chk_use_processed.isChecked() and self.processed_image is not None) else self.current_image
        if img is None:
            QMessageBox.information(self, "Info", "Load an image first.")
            return
        size = (int(self.spn_in_w.value()), int(self.spn_in_h.value()))
        gray = self.chk_gray_in.isChecked()
        norm01 = self.chk_norm01.isChecked()
        p = self.clf.predict_image(img, size=size, grayscale=gray, norm01=norm01)
        thr = float(self.spn_thresh.value())
        label = "Real" if p >= thr else "Fake"
        self.lbl_pred.setText(f"Prediction: {label} ({p:.3f})")
        self.last_prediction = (label, float(p))
        self._log(f"Predicted: {label} prob={p:.3f} thr={thr}")

    def _populate_table(self, feats: dict):
        self.tbl.setRowCount(0)
        for i, (k, v) in enumerate(feats.items()):
            self.tbl.insertRow(i)
            self.tbl.setItem(i, 0, QTableWidgetItem(str(k)))
            self.tbl.setItem(i, 1, QTableWidgetItem(f"{float(v):.6f}"))

    def _log(self, msg: str):
        if hasattr(self, "txt_results"):
            self.txt_results.append(msg)

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
