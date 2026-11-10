import warnings
warnings.filterwarnings('ignore')
import gc
import math
import os
import queue
import sys
import threading
import time
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image, ImageDraw, ImageFont
from PySide6 import QtCore, QtGui, QtWidgets
Signal = QtCore.Signal
from config import args as parsed_args
from config import return_args
from image import load_data
from Networks.CDETR import build_model
import util.misc as utils
from utils import setup_seed

CHECKPOINT_PATH = 'saved_suwon/model_best.pth'
TEST_LIST_PATH = 'npydata/suwon_test.npy'
SAVE_DIR = 'saved_suwon'
SUBMISSION_PATH = os.path.join(SAVE_DIR, 'submission.csv')
METRICS_PATH = os.path.join(SAVE_DIR, 'metrics.csv')
OUTPUT_VIDEO_PATH = 'output.mp4'
OUTPUT_VIDEO_FPS = 25
THRESHOLD = 0.24 # 0.265
NUM_QUERIES = 700
CROP_SIZE = 256
GPU_IDS = '0'
USE_FP16 = True
TARGET_FPS_OPTIONS = [5, 10, 20]
EXACT_FPS_OPTIONS = [5, 10]
FAST_CUDNN_BENCHMARK = True
QUEUE_SIZE = 8
CLOSING_SECONDS = 10
LOG_PANEL_LINES = 13
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

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

def save_csv(path, headers, rows):
    with open(path, 'w', encoding='utf-8') as file:
        file.write(','.join(headers) + '\n')
        for row in rows:
            file.write(','.join(str(value) for value in row) + '\n')

class ChartRenderer:
    def __init__(self, total_frames, chart_width, chart_height):
        self.total_frames = total_frames
        self.chart_width = chart_width
        self.chart_height = chart_height
        self.pred_counts = []
        self.gt_counts = []
        self.y_max = None
        self.chart_img = None
        self.transform = None
        self.canvas_height = None

    def draw_background(self):
        fig, ax = plt.subplots(figsize=(self.chart_width / 100, self.chart_height / 100), dpi=100)
        canvas = FigureCanvas(fig)
        y_max = self.y_max
        step_v = y_max / 5
        v_ticks = [int(step_v * i) for i in range(6)]
        step_h = max(self.total_frames, 1) / 5
        h_ticks = [int(step_h * i) for i in range(6)]
        ax.axhspan(0, 24, facecolor='green', alpha=0.15, label='Low (0 - 24)')
        ax.axhspan(25, 100, facecolor='yellow', alpha=0.15, label='Medium (25 - 100)')
        ax.axhspan(101, y_max, facecolor='red', alpha=0.15, label='High (> 100)')
        ax.plot([], [], color='black', linewidth=2, label='Pred Count')
        ax.plot([], [], color='blue', linewidth=2, label='GT Count')
        ax.set_xlim(0, max(self.total_frames, 1))
        ax.set_ylim(0, y_max)
        ax.set_xticks(h_ticks)
        ax.set_yticks(v_ticks)
        ax.set_xticklabels([''] + [str(tick) for tick in h_ticks[1:]])
        ax.set_xlabel('Frame', fontsize=26)
        ax.set_ylabel('Count', fontsize=26)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        legend = ax.legend(loc='upper right', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        legend.set_visible(False)
        canvas.draw()
        chart_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        self.chart_img = chart_img.reshape(canvas.get_width_height()[::-1] + (3,)).copy()
        self.canvas_height = self.chart_img.shape[0]
        self.transform = ax.transData
        legend.set_visible(True)
        for artist in ax.get_children():
            if artist is not legend:
                artist.set_visible(False)
        fig.patch.set_alpha(0)
        canvas.draw()
        legend_rgba = np.asarray(canvas.buffer_rgba()).astype(np.float32)
        ys, xs = np.nonzero(legend_rgba[..., 3] > 0)
        self.legend_box = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
        legend_rgba = legend_rgba[self.legend_box[0]:self.legend_box[1], self.legend_box[2]:self.legend_box[3]]
        self.legend_rgb = legend_rgba[..., :3]
        self.legend_alpha = legend_rgba[..., 3:] / 255.0
        plt.close(fig)
        for idx in range(1, len(self.pred_counts)):
            self.draw_segment(idx)

    def to_pixel(self, x, y):
        px, py = self.transform.transform((x, y))
        return int(round(px)), int(round(self.canvas_height - py))

    def draw_segment(self, idx):
        cv2.line(self.chart_img, self.to_pixel(idx - 1, self.pred_counts[idx - 1]), self.to_pixel(idx, self.pred_counts[idx]), (0, 0, 0), 3, cv2.LINE_AA)
        cv2.line(self.chart_img, self.to_pixel(idx - 1, self.gt_counts[idx - 1]), self.to_pixel(idx, self.gt_counts[idx]), (0, 0, 255), 3, cv2.LINE_AA)

    def prepare(self):
        self.y_max = max(max(self.pred_counts + self.gt_counts + [1]), 110)
        self.draw_background()

    def update(self, pred_count, gt_count):
        self.pred_counts.append(pred_count)
        self.gt_counts.append(gt_count)
        y_max = max(max(self.pred_counts + self.gt_counts + [1]), 110)
        if y_max != self.y_max:
            self.y_max = y_max
            self.draw_background()
        elif len(self.pred_counts) > 1:
            self.draw_segment(len(self.pred_counts) - 1)
        chart_img = self.chart_img.copy()
        y0, y1, x0, x1 = self.legend_box
        region = chart_img[y0:y1, x0:x1].astype(np.float32)
        chart_img[y0:y1, x0:x1] = np.clip(self.legend_rgb * self.legend_alpha + region * (1.0 - self.legend_alpha) + 0.5, 0, 255).astype(np.uint8)
        if chart_img.shape[:2] != (self.chart_height, self.chart_width):
            chart_img = cv2.resize(chart_img, (self.chart_width, self.chart_height))
        return chart_img

def image_to_patches(frame_rgb, mean, std):
    img = torch.from_numpy(frame_rgb).cuda(non_blocking=True).permute(2, 0, 1).float().div(255)
    img = (img - mean) / std
    padding_h = img.shape[1] % CROP_SIZE
    padding_w = img.shape[2] % CROP_SIZE
    if padding_w != 0:
        padding_w = CROP_SIZE - padding_w
    if padding_h != 0:
        padding_h = CROP_SIZE - padding_h
    img = F.pad(img, (padding_w, 0, padding_h, 0), 'constant')
    width, height = img.shape[2], img.shape[1]
    num_w = int(width / CROP_SIZE)
    num_h = int(height / CROP_SIZE)
    img_return = img.view(3, num_h, CROP_SIZE, width).view(3, num_h, CROP_SIZE, num_w, CROP_SIZE)
    img_return = img_return.permute(0, 1, 3, 2, 4).contiguous().view(3, num_w * num_h, CROP_SIZE, CROP_SIZE).permute(1, 0, 2, 3)
    patch_info = [num_h, num_w, height, width, CROP_SIZE, padding_w, padding_h]
    return img_return, patch_info

def prediction_map_from_outputs(outputs, patch_info, threshold, h, w):
    out_logits = outputs['pred_logits']
    out_points = outputs['pred_points']
    prob = out_logits.sigmoid()
    num_patches = out_logits.shape[0]
    num_classes = out_logits.shape[2]
    num_h, num_w, padded_h, padded_w, crop_size, padding_w, padding_h = patch_info
    topk_values, topk_indexes = torch.topk(prob.reshape(num_patches, -1), NUM_QUERIES, dim=1)
    point_indexes = torch.div(topk_indexes, num_classes, rounding_mode='floor')
    patch_points = torch.gather(out_points[:, :, :2], 1, point_indexes.unsqueeze(-1).expand(-1, -1, 2)) * crop_size
    values = topk_values.detach().float().cpu().numpy()
    points = patch_points.detach().float().cpu().numpy()
    keep = values >= threshold
    y = np.clip(points[..., 0], 0, crop_size - 1).astype(np.int64)
    x = np.clip(points[..., 1], 0, crop_size - 1).astype(np.int64)
    y = (y + (np.arange(num_patches) // num_w)[:, None] * crop_size - padding_h)[keep]
    x = (x + (np.arange(num_patches) % num_w)[:, None] * crop_size - padding_w)[keep]
    valid = (y >= 0) & (x >= 0) & (y < h) & (x < w)
    kpoint = np.zeros((h, w), dtype=np.uint8)
    kpoint[y[valid], x[valid]] = 1
    return kpoint

def build_visual_frame(frame_rgb, pred_kpoint, chart_img):
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
    top_row = np.hstack((overlay, density))
    return np.vstack((top_row, chart_img))

class VideoPanel:
    def __init__(self, width):
        scale = width / 2560.0
        self.width = width
        self.log_height = int(round(414 * scale))
        self.label_height = int(round(92 * scale))
        self.line_height = int(round(29.6 * scale))
        self.margin = int(round(10 * scale))
        self.font = ImageFont.truetype(os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', 'DejaVuSans.ttf'), int(round(22 * scale)))
        background = Image.new('RGB', (self.width, self.log_height + self.label_height), (239, 239, 239))
        ImageDraw.Draw(background).rectangle([0, 0, self.width - 1, self.log_height - 1], fill=(255, 255, 255), outline=(160, 160, 160))
        self.background = np.asarray(background)
        self.line_cache = {}
        self.label_cache = {}

    def render_line(self, line):
        strip = Image.new('RGB', (self.width - 2 * self.margin, self.line_height), (255, 255, 255))
        ImageDraw.Draw(strip).text((0, 0), line, font=self.font, fill=(0, 0, 0))
        return np.asarray(strip)

    def render_label(self, label_text):
        strip = Image.new('RGB', (self.width, self.label_height), (239, 239, 239))
        draw = ImageDraw.Draw(strip)
        text_width = draw.textlength(label_text, font=self.font)
        draw.text(((self.width - text_width) / 2, self.label_height / 2), label_text, font=self.font, fill=(0, 0, 0), anchor='lm')
        return np.asarray(strip)

    def render(self, lines, label_text):
        lines = list(lines[-LOG_PANEL_LINES:])
        self.line_cache = {line: self.line_cache[line] if line in self.line_cache else self.render_line(line) for line in lines}
        if label_text not in self.label_cache:
            self.label_cache[label_text] = self.render_label(label_text)
        panel = self.background.copy()
        for row, line in enumerate(lines):
            top = self.margin + row * self.line_height
            panel[top:top + self.line_height, self.margin:self.width - self.margin] = self.line_cache[line]
        panel[self.log_height:] = self.label_cache[label_text]
        return panel

def write_video_frames(writer, frame_bgr, written, video_time, video_fps):
    target = int(round(video_time * video_fps))
    while written < target:
        writer.write(frame_bgr)
        written += 1
    return written

def put_item(item_queue, item, stop_event):
    while not stop_event.is_set():
        try:
            item_queue.put(item, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False

def get_item(item_queue, stop_event):
    while not stop_event.is_set():
        try:
            return item_queue.get(timeout=0.1)
        except queue.Empty:
            continue
    return None

class InferenceWorker(QtCore.QObject):
    log = Signal(str)
    frame = Signal(object)
    finished = Signal(float, float, str)
    error = Signal(str)

    def __init__(self, target_fps):
        super().__init__()
        self.target_fps = target_fps

    @QtCore.Slot()
    def run(self):
        result = None
        message = None
        try:
            result = self.run_testing()
        except Exception as exc:
            message = str(exc)
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()
        if result is None:
            self.error.emit(message)
        else:
            self.finished.emit(*result)

    def run_testing(self):
        configure_args()
        self.log.emit('Loading test set: {}'.format(TEST_LIST_PATH))
        test_list = np.load(TEST_LIST_PATH, allow_pickle=True).tolist()
        total_frames = len(test_list)
        self.log.emit('Loading model: {}'.format(CHECKPOINT_PATH))
        utils.init_distributed_mode(return_args)
        model, _, _ = build_model(return_args)
        model = model.cuda()
        checkpoint = torch.load(CHECKPOINT_PATH)
        nn.DataParallel(model, device_ids=[0]).load_state_dict(checkpoint['state_dict'], strict=False)
        model.eval()
        exact_mode = self.target_fps in EXACT_FPS_OPTIONS
        memory_format = torch.contiguous_format if exact_mode else torch.channels_last
        model = model.to(memory_format=memory_format)
        torch.backends.cudnn.benchmark = (not exact_mode) and FAST_CUDNN_BENCHMARK
        torch.backends.cudnn.deterministic = not torch.backends.cudnn.benchmark
        inference_model = model if exact_mode else torch.compile(model, mode='reduce-overhead')
        mean = torch.tensor(IMAGE_MEAN, device='cuda').view(3, 1, 1)
        std = torch.tensor(IMAGE_STD, device='cuda').view(3, 1, 1)
        warmup_img, _ = load_data(test_list[0])
        warmup_frame = np.asarray(warmup_img).copy()
        warmup_patches, _ = image_to_patches(warmup_frame, mean, std)
        with torch.no_grad():
            for _ in range(3):
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    inference_model(warmup_patches.contiguous(memory_format=memory_format))
        torch.cuda.synchronize()
        chart = ChartRenderer(total_frames, warmup_frame.shape[1] * 2, warmup_frame.shape[0])
        chart.prepare()
        read_queue = queue.Queue(maxsize=QUEUE_SIZE)
        visual_queue = queue.Queue(maxsize=QUEUE_SIZE)
        write_queue = queue.Queue(maxsize=QUEUE_SIZE)
        video_queue = queue.Queue()
        stop_event = threading.Event()
        errors = []
        submission_rows = []
        state = {'mae': 0.0, 'rmse_accum': 0.0, 'correct': 0, 'total': 0, 'end_time': None}
        frame_interval = 1.0 / self.target_fps
        start_time = time.time()

        def read_frames():
            try:
                for image_path in test_list:
                    img_pil, kpoint = load_data(image_path)
                    item = (os.path.basename(image_path), np.asarray(img_pil).copy(), float(np.sum(kpoint)))
                    if not put_item(read_queue, item, stop_event):
                        return
                put_item(read_queue, None, stop_event)
            except Exception as exc:
                errors.append(exc)
                stop_event.set()

        def visualize_frames():
            try:
                while True:
                    item = get_item(visual_queue, stop_event)
                    if item is None:
                        put_item(write_queue, None, stop_event)
                        return
                    idx, fname, frame_rgb, pred_map, count, gt_count = item
                    state['mae'] += abs(count - gt_count)
                    state['rmse_accum'] += abs(count - gt_count) * abs(count - gt_count)
                    state['correct'] += int(get_congestion_label(count) == get_congestion_label(gt_count))
                    state['total'] += 1
                    submission_rows.append([fname, float(count), float(gt_count)])
                    visual_rgb = build_visual_frame(frame_rgb, pred_map, chart.update(float(count), float(gt_count)))
                    if not put_item(write_queue, visual_rgb, stop_event):
                        return
                    next_time = (state['end_time'] if state['end_time'] is not None else start_time) + frame_interval
                    slot_time = start_time + (idx + 1) * frame_interval
                    if next_time - slot_time > frame_interval / 2:
                        next_time = slot_time
                    wait_time = next_time - time.time()
                    if wait_time > 0:
                        time.sleep(wait_time)
                    self.frame.emit(visual_rgb)
                    state['end_time'] = time.time()
                    elapsed = state['end_time'] - start_time
                    fps = state['total'] / elapsed if elapsed > 0 else 0.0
                    info = '[{}/{}], Frame: {}, GT Count: {:.1f}, Pred Count: {:.1f}'.format(idx + 1, total_frames, fname, gt_count, count)
                    self.log.emit('{}, FPS: {:.2f}'.format(info, fps))
                    video_queue.put((elapsed, '{}, FPS: {:.2f}'.format(info, fps)))
            except Exception as exc:
                errors.append(exc)
                stop_event.set()

        def write_frames():
            writer = None
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                panel = None
                frame_bgr = None
                written = 0
                lines = ['Loading test set: {}'.format(TEST_LIST_PATH), 'Loading model: {}'.format(CHECKPOINT_PATH)]
                while True:
                    visual_rgb = get_item(write_queue, stop_event)
                    if visual_rgb is None:
                        break
                    video_item = get_item(video_queue, stop_event)
                    if video_item is None:
                        return
                    display_time, line = video_item
                    lines.append(line)
                    vh, vw = visual_rgb.shape[:2]
                    if writer is None:
                        panel = VideoPanel(vw)
                        writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, self.target_fps, (vw, vh + panel.log_height + panel.label_height))
                        frame_bgr = np.empty((vh + panel.log_height + panel.label_height, vw, 3), dtype=np.uint8)
                    else:
                        written = write_video_frames(writer, frame_bgr, written, display_time, self.target_fps)
                    cv2.cvtColor(visual_rgb, cv2.COLOR_RGB2BGR, dst=frame_bgr[:vh])
                    cv2.cvtColor(panel.render(lines, 'Testing is running...'), cv2.COLOR_RGB2BGR, dst=frame_bgr[vh:])
                if writer is not None and not stop_event.is_set():
                    total = state['total']
                    elapsed = state['end_time'] - start_time
                    average_fps = total / elapsed if elapsed > 0 else 0.0
                    accuracy = state['correct'] / max(total, 1) * 100
                    written = write_video_frames(writer, frame_bgr, written, elapsed, self.target_fps)
                    lines += ['Average FPS: {:.2f}, Accuracy: {:.2f}%'.format(average_fps, accuracy), 'Saved submission: {}'.format(SUBMISSION_PATH), 'Saved metrics: {}'.format(METRICS_PATH), 'Saved output video: {}'.format(OUTPUT_VIDEO_PATH)]
                    cv2.cvtColor(panel.render(lines, 'Average FPS: {:.2f}    Accuracy: {:.2f}%'.format(average_fps, accuracy)), cv2.COLOR_RGB2BGR, dst=frame_bgr[vh:])
                    for _ in range(int(CLOSING_SECONDS * self.target_fps)):
                        writer.write(frame_bgr)
            except Exception as exc:
                errors.append(exc)
                stop_event.set()
            finally:
                if writer is not None:
                    writer.release()

        threads = [threading.Thread(target=target, daemon=True) for target in (read_frames, visualize_frames, write_frames)]
        for thread in threads:
            thread.start()
        try:
            with torch.no_grad():
                for idx in range(total_frames + 1):
                    item = get_item(read_queue, stop_event)
                    if item is None:
                        put_item(visual_queue, None, stop_event)
                        break
                    fname, frame_rgb, gt_count = item
                    img, patch_info = image_to_patches(frame_rgb, mean, std)
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        outputs = inference_model(img.contiguous(memory_format=memory_format))
                    out_logits = outputs['pred_logits']
                    prob = out_logits.sigmoid()
                    prob = prob.view(1, -1, 2)
                    topk_values, _ = torch.topk(prob.view(1, -1), img.shape[0] * NUM_QUERIES, dim=1)
                    count = 0
                    for row in range(topk_values.shape[0]):
                        sub_count = topk_values[row, :]
                        sub_count = (sub_count >= THRESHOLD).float().sum().item()
                        count += sub_count
                    pred_map = prediction_map_from_outputs(outputs, patch_info, THRESHOLD, frame_rgb.shape[0], frame_rgb.shape[1])
                    if not put_item(visual_queue, (idx, fname, frame_rgb, pred_map, count, gt_count), stop_event):
                        break
        except Exception as exc:
            errors.append(exc)
            stop_event.set()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]
        total = state['total']
        elapsed = state['end_time'] - start_time if state['end_time'] is not None else 0.0
        average_fps = total / elapsed if elapsed > 0 else 0.0
        final_mae = state['mae'] / max(total, 1)
        final_rmse = math.sqrt(state['rmse_accum'] / max(total, 1))
        accuracy = state['correct'] / max(total, 1) * 100
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_csv(SUBMISSION_PATH, ['path', 'total_count', 'gt_count'], submission_rows)
        save_csv(METRICS_PATH, ['MAE', 'RMSE', 'Accuracy'], [[f'{final_mae:.2f}', f'{final_rmse:.2f}', f'{accuracy:.2f}']])
        self.log.emit('Average FPS: {:.2f}, Accuracy: {:.2f}%'.format(average_fps, accuracy))
        self.log.emit('Saved submission: {}'.format(SUBMISSION_PATH))
        self.log.emit('Saved metrics: {}'.format(METRICS_PATH))
        return average_fps, accuracy, OUTPUT_VIDEO_PATH

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
        self.setWindowTitle('Traffic Congestion Detection')
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
        self.run_buttons = []
        run_layout = QtWidgets.QHBoxLayout()
        for target_fps in TARGET_FPS_OPTIONS:
            run_button = QtWidgets.QPushButton('Run Testing ({} FPS)'.format(target_fps))
            run_button.setMinimumHeight(44)
            run_button.clicked.connect(lambda checked=False, fps=target_fps: self.start_inference(fps))
            run_layout.addWidget(run_button)
            self.run_buttons.append(run_button)
        self.image_label = ClickableVideoLabel('Click Run Testing button to start')
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(960, 540)
        self.image_label.setStyleSheet('background-color: #111; color: white;')
        self.image_label.clicked.connect(self.toggle_video_playback)
        self.play_button = QtWidgets.QPushButton('Start')
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.toggle_video_playback)
        self.video_slider = ClickableSlider(QtCore.Qt.Orientation.Horizontal)
        self.video_slider.setEnabled(False)
        self.video_slider.setRange(0, 0)
        self.video_slider.valueChanged.connect(self.seek_video)
        self.log_box = QtWidgets.QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(160)
        self.metrics_label = QtWidgets.QLabel('The metrics will appear here after testing finishes.')
        self.metrics_label.setMinimumHeight(36)
        self.metrics_label.setAlignment(QtCore.Qt.AlignCenter)
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(run_layout)
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

    def set_run_buttons_enabled(self, enabled):
        for run_button in self.run_buttons:
            run_button.setEnabled(enabled)

    def start_inference(self, target_fps):
        self.set_run_buttons_enabled(False)
        self.log_box.clear()
        self.metrics_label.setText('Testing is running...')
        self.video_timer.stop()
        self.video_is_playing = False
        self.play_button.setEnabled(False)
        self.play_button.setText('Start')
        self.video_slider.setEnabled(False)
        self.video_slider.setRange(0, 0)
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self.thread = QtCore.QThread(self)
        self.worker = InferenceWorker(target_fps)
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

    def on_finished(self, average_fps, accuracy, output_path):
        self.metrics_label.setText('Average FPS: {:.2f}    Accuracy: {:.2f}%'.format(average_fps, accuracy))
        self.append_log('Saved output video: {}'.format(output_path))
        self.set_run_buttons_enabled(True)
        self.start_video_playback(output_path)

    def on_error(self, message):
        self.append_log('ERROR: {}'.format(message))
        self.set_run_buttons_enabled(True)

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
        self.play_button.setText('Stop')
        self.video_is_playing = True
        self.play_next_frame()
        self.video_timer.start(max(int(1000 / self.video_fps), 1))

    def toggle_video_playback(self):
        if self.video_capture is None:
            return
        if self.video_is_playing:
            self.video_timer.stop()
            self.video_is_playing = False
            self.play_button.setText('Start')
        else:
            self.video_is_playing = True
            self.play_button.setText('Stop')
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
