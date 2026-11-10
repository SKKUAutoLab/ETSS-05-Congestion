import warnings
warnings.filterwarnings('ignore')
import math
import os
import sys
import time
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torchvision import transforms
from PySide6 import QtCore, QtGui, QtWidgets
Signal = QtCore.Signal
from config import args as parsed_args
from config import return_args
import dataset
from image import load_data
from Networks.CDETR import build_model
import util.misc as utils
from utils import setup_seed

CHECKPOINT_PATH = 'saved_suwon/checkpoint.pth'
TEST_LIST_PATH = 'npydata/suwon_test.npy'
OUTPUT_VIDEO_PATH = 'output.mp4'
OUTPUT_VIDEO_FPS = 25
THRESHOLD = 0.25
NUM_QUERIES = 700
CROP_SIZE = 256
GPU_IDS = '0'
USE_FP16 = True

def configure_args():
    fixed_values = {'type_dataset': 'suwon', 'pre': CHECKPOINT_PATH, 'threshold': THRESHOLD, 'num_queries': NUM_QUERIES, 'crop_size': CROP_SIZE, 'gpu_id': GPU_IDS,
                    'fp16': USE_FP16, 'save': False}
    for key, value in fixed_values.items():
        setattr(parsed_args, key, value)
        setattr(return_args, key, value)
    return vars(parsed_args)

def get_congestion_label(count):
    if count <= 24:
        return 0
    if count <= 100:
        return 1
    return 2

def generate_chart_image(pred_counts, gt_counts, total_frames, chart_width, chart_height):
    fig, ax = plt.subplots(figsize=(chart_width / 100, chart_height / 100), dpi=100)
    canvas = FigureCanvas(fig)
    max_count = max(pred_counts + gt_counts + [1])
    y_max = max(max_count, 110)
    step_v = y_max / 5
    v_ticks = [int(step_v * i) for i in range(6)]
    step_h = max(total_frames, 1) / 5
    h_ticks = [int(step_h * i) for i in range(6)]
    ax.axhspan(0, 24, facecolor='green', alpha=0.15, label='Low (0-24)')
    ax.axhspan(25, 100, facecolor='yellow', alpha=0.15, label='Medium (25-100)')
    ax.axhspan(101, y_max, facecolor='red', alpha=0.15, label='High (>100)')
    ax.plot(range(len(pred_counts)), pred_counts, color='black', linewidth=2, label='Pred')
    ax.plot(range(len(gt_counts)), gt_counts, color='blue', linewidth=2, label='GT')
    ax.set_xlim(0, max(total_frames, 1))
    ax.set_ylim(0, y_max)
    ax.set_xticks(h_ticks)
    ax.set_yticks(v_ticks)
    ax.set_xlabel('Frame', fontsize=26)
    ax.set_ylabel('Count', fontsize=26)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.legend(loc='upper right', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    canvas.draw()
    chart_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    chart_img = chart_img.reshape(canvas.get_width_height()[::-1] + (3,))
    chart_img = cv2.resize(chart_img, (chart_width, chart_height))
    plt.close(fig)
    return chart_img

def prediction_map_from_outputs(outputs, patch_info, threshold):
    out_logits = outputs['pred_logits']
    out_points = outputs['pred_points']
    prob = out_logits.sigmoid()
    num_patches = out_logits.shape[0]
    num_classes = out_logits.shape[2]
    num_h, num_w, padded_h, padded_w, crop_size, padding_w, padding_h = patch_info
    kpoint_list = []
    for patch_idx in range(num_patches):
        patch_prob = prob[patch_idx].reshape(-1)
        topk_values, topk_indexes = torch.topk(patch_prob, NUM_QUERIES, dim=0)
        point_indexes = torch.div(topk_indexes, num_classes, rounding_mode='floor')
        patch_points = out_points[patch_idx, point_indexes, :2] * crop_size
        patch_map = np.zeros((crop_size, crop_size), dtype=np.uint8)
        values = topk_values.detach().float().cpu().numpy()
        points = patch_points.detach().float().cpu().numpy()
        for value, point in zip(values, points):
            if value < threshold:
                continue
            y = int(np.clip(point[0], 0, crop_size - 1))
            x = int(np.clip(point[1], 0, crop_size - 1))
            patch_map[y, x] = 1
        kpoint_list.append(patch_map)
    kpoint = torch.from_numpy(np.array(kpoint_list)).unsqueeze(0)
    kpoint = kpoint.view(num_h, num_w, crop_size, crop_size)
    kpoint = kpoint.permute(0, 2, 1, 3).contiguous()
    kpoint = kpoint.view(num_h, crop_size, padded_w).view(padded_h, padded_w).numpy()
    if padding_h > 0:
        kpoint = kpoint[padding_h:, :]
    if padding_w > 0:
        kpoint = kpoint[:, padding_w:]
    return kpoint

def build_visual_frame(image_path, pred_kpoint, pred_counts, gt_counts, total_frames):
    img_pil, _ = load_data(image_path)
    frame_rgb = np.asarray(img_pil).copy()
    h, w = frame_rgb.shape[:2]
    if pred_kpoint.shape[:2] != (h, w):
        pred_kpoint = pred_kpoint[:h, :w]
    overlay = frame_rgb.copy()
    pred_y, pred_x = np.nonzero(pred_kpoint)
    for x, y in zip(pred_x, pred_y):
        cv2.circle(overlay, (int(x), int(y)), 3, (0, 255, 50), -1)
    if pred_kpoint.max() > 0:
        density = cv2.GaussianBlur(pred_kpoint.astype(np.float32), (0, 0), 6)
        density = density / max(float(density.max()), 1e-6) * 255
    else:
        density = np.zeros_like(pred_kpoint, dtype=np.float32)
    density = density.astype(np.uint8)
    density = cv2.applyColorMap(density, 2)
    density = cv2.cvtColor(density, cv2.COLOR_BGR2RGB)
    chart_img = generate_chart_image(pred_counts, gt_counts, total_frames, w * 2, h)
    top_row = np.hstack((overlay, density))
    return np.vstack((top_row, chart_img))

class InferenceWorker(QtCore.QObject):
    log = Signal(str)
    frame = Signal(object)
    finished = Signal(float, float, float, str)
    error = Signal(str)

    @QtCore.Slot()
    def run(self):
        try:
            params = configure_args()
            self.log.emit('Loading test set: {}'.format(TEST_LIST_PATH))
            test_list = np.load(TEST_LIST_PATH, allow_pickle=True).tolist()
            path_by_name = {os.path.basename(path): path for path in test_list}
            self.log.emit('Loading model: {}'.format(CHECKPOINT_PATH))
            utils.init_distributed_mode(return_args)
            model, _, _ = build_model(return_args)
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=[0])
            checkpoint = torch.load(CHECKPOINT_PATH)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            model.eval()
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            test_data = dataset.listDataset(test_list, params['output_dir'], shuffle=False, transform=transform, args=params, train=False)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = None
            pred_counts = []
            gt_counts = []
            mae = 0.0
            rmse_accum = 0.0
            correct = 0
            total = 0
            start_time = time.time()
            for idx, (fname, img, kpoint, targets, patch_info) in enumerate(test_loader):
                if len(img.shape) == 5:
                    img = img.squeeze(0)
                if len(img.shape) == 3:
                    img = img.unsqueeze(0)
                if len(kpoint.shape) == 5:
                    kpoint = kpoint.squeeze(0)
                with torch.no_grad():
                    img = img.cuda()
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        outputs = model(img)
                out_logits = outputs['pred_logits']
                prob = out_logits.sigmoid()
                prob = prob.view(1, -1, 2)
                topk_values, _ = torch.topk(prob.view(1, -1), kpoint.shape[0] * NUM_QUERIES, dim=1)
                count = 0
                for row in range(topk_values.shape[0]):
                    sub_count = topk_values[row, :]
                    sub_count = (sub_count >= THRESHOLD).float().sum().item()
                    count += sub_count
                gt_count = torch.sum(kpoint).item()
                pred_counts.append(float(count))
                gt_counts.append(float(gt_count))
                mae += abs(count - gt_count)
                rmse_accum += abs(count - gt_count) * abs(count - gt_count)
                correct += int(get_congestion_label(count) == get_congestion_label(gt_count))
                total += 1
                info = 'Name: {}, GT: {:.4f}, Pred: {:.4f}'.format(fname[0], gt_count, count)
                elapsed = time.time() - start_time
                fps = total / elapsed if elapsed > 0 else 0.0
                self.log.emit('{} | FPS: {:.2f}'.format(info, fps))
                image_path = path_by_name[fname[0]]
                patch_info_values = [int(v.item()) if torch.is_tensor(v) else int(v) for v in patch_info]
                pred_map = prediction_map_from_outputs(outputs, patch_info_values, THRESHOLD)
                visual_rgb = build_visual_frame(image_path, pred_map, pred_counts, gt_counts, len(test_loader))
                if writer is None:
                    vh, vw = visual_rgb.shape[:2]
                    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, OUTPUT_VIDEO_FPS, (vw, vh))
                writer.write(cv2.cvtColor(visual_rgb, cv2.COLOR_RGB2BGR))
                self.frame.emit(visual_rgb)
            if writer is not None:
                writer.release()
            final_mae = mae / max(total, 1)
            final_rmse = math.sqrt(rmse_accum / max(total, 1))
            accuracy = correct / max(total, 1) * 100
            self.log.emit('MAE: {:.4f}, RMSE: {:.4f}, Accuracy: {:.2f}%'.format(final_mae, final_rmse, accuracy))
            self.finished.emit(final_mae, final_rmse, accuracy, OUTPUT_VIDEO_PATH)
        except Exception as exc:
            self.error.emit(str(exc))

class ClickableVideoLabel(QtWidgets.QLabel):
    clicked = Signal()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

class ClickableSlider(QtWidgets.QSlider):
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self.maximum() > self.minimum():
            ratio = event.position().x() / max(self.width(), 1)
            value = self.minimum() + round(ratio * (self.maximum() - self.minimum()))
            self.setValue(int(np.clip(value, self.minimum(), self.maximum())))
        super().mousePressEvent(event)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Suwon Traffic Congestion Demo')
        self.resize(1200, 900)
        self.thread = None
        self.worker = None
        self.video_capture = None
        self.video_frame_count = 0
        self.video_fps = float(OUTPUT_VIDEO_FPS)
        self.video_is_playing = False
        self.slider_is_updating = False
        self.video_timer = QtCore.QTimer(self)
        self.video_timer.timeout.connect(self.play_next_frame)
        self.run_button = QtWidgets.QPushButton('Run')
        self.run_button.setMinimumHeight(44)
        self.run_button.clicked.connect(self.start_inference)
        self.image_label = ClickableVideoLabel('Click Run to start')
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(960, 540)
        self.image_label.setStyleSheet('background-color: #111; color: white;')
        self.image_label.clicked.connect(self.toggle_video_playback)
        self.play_button = QtWidgets.QPushButton('Play')
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.toggle_video_playback)
        self.video_slider = ClickableSlider(QtCore.Qt.Orientation.Horizontal)
        self.video_slider.setEnabled(False)
        self.video_slider.setRange(0, 0)
        self.video_slider.valueChanged.connect(self.seek_video)
        self.log_box = QtWidgets.QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(160)
        self.metrics_label = QtWidgets.QLabel('MAE: -    RMSE: -    Accuracy: -')
        self.metrics_label.setMinimumHeight(36)
        self.metrics_label.setAlignment(QtCore.Qt.AlignCenter)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.run_button)
        layout.addWidget(self.image_label, stretch=1)
        playback_layout = QtWidgets.QHBoxLayout()
        playback_layout.addWidget(self.play_button)
        playback_layout.addWidget(self.video_slider, stretch=1)
        layout.addLayout(playback_layout)
        layout.addWidget(self.log_box)
        layout.addWidget(self.metrics_label)
        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def append_log(self, text):
        self.log_box.append(text)

    def start_inference(self):
        self.run_button.setEnabled(False)
        self.log_box.clear()
        self.metrics_label.setText('MAE: -    RMSE: -    Accuracy: -')
        self.video_timer.stop()
        self.video_is_playing = False
        self.play_button.setEnabled(False)
        self.play_button.setText('Play')
        self.video_slider.setEnabled(False)
        self.video_slider.setRange(0, 0)
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self.thread = QtCore.QThread(self)
        self.worker = InferenceWorker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.append_log)
        self.worker.frame.connect(self.show_numpy_frame)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def show_numpy_frame(self, frame_rgb):
        h, w = frame_rgb.shape[:2]
        bytes_per_line = 3 * w
        qimage = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()
        pixmap = QtGui.QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

    def on_finished(self, mae, rmse, accuracy, output_path):
        self.metrics_label.setText('MAE: {:.4f}    RMSE: {:.4f}    Accuracy: {:.2f}%'.format(mae, rmse, accuracy))
        self.append_log('Saved output video: {}'.format(output_path))
        self.run_button.setEnabled(True)
        self.start_video_playback(output_path)

    def on_error(self, message):
        self.append_log('ERROR: {}'.format(message))
        self.run_button.setEnabled(True)

    def start_video_playback(self, output_path):
        self.video_timer.stop()
        self.video_is_playing = False
        if self.video_capture is not None:
            self.video_capture.release()
        self.video_capture = cv2.VideoCapture(output_path)
        if not self.video_capture.isOpened():
            self.append_log('Could not open output video for playback')
            return
        self.video_frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        if self.video_fps <= 0:
            self.video_fps = float(OUTPUT_VIDEO_FPS)
        self.slider_is_updating = True
        self.video_slider.setRange(0, max(self.video_frame_count - 1, 0))
        self.video_slider.setValue(0)
        self.slider_is_updating = False
        self.video_slider.setEnabled(self.video_frame_count > 0)
        self.play_button.setEnabled(True)
        self.play_button.setText('Pause')
        self.video_is_playing = True
        self.play_next_frame()
        self.video_timer.start(max(int(1000 / self.video_fps), 1))

    def toggle_video_playback(self):
        if self.video_capture is None:
            return
        if self.video_is_playing:
            self.video_timer.stop()
            self.video_is_playing = False
            self.play_button.setText('Play')
        else:
            self.video_is_playing = True
            self.play_button.setText('Pause')
            self.video_timer.start(max(int(1000 / self.video_fps), 1))

    def seek_video(self, frame_index):
        if self.slider_is_updating or self.video_capture is None:
            return
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame_bgr = self.video_capture.read()
        if not ret:
            return
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.show_numpy_frame(frame_rgb)
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    def play_next_frame(self):
        if self.video_capture is None:
            return
        ret, frame_bgr = self.video_capture.read()
        if not ret:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.slider_is_updating = True
            self.video_slider.setValue(0)
            self.slider_is_updating = False
            return
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.show_numpy_frame(frame_rgb)
        current_frame = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        self.slider_is_updating = True
        self.video_slider.setValue(max(current_frame, 0))
        self.slider_is_updating = False

def main():
    setup_seed(parsed_args.seed)
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()