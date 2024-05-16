import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import subprocess
import tensorflow as tf2
import time
import numpy as np
import argparse

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

SAVED_MODEL_PATH = r"./model"

logger = tf2.get_logger()

# Detected classes by the model
CLASSES = {
    '1': {
        'label': 'Container',
        'color': [0, 255, 0],
        'focused_color': [0, 0, 255]
    },
    '2': {
        'label': 'Spreader',
        'color': [255, 0, 0],
        'focused_color': [0, 0, 255]
    },
    '3': {
        'label': 'Truck',
        'color': [255, 255, 0],
        'focused_color': [0, 0, 255]
    }
}


def load_model():
    # cargamos el modelo
    logger.info('Loading model... ')
    start_time = time.time()

    # Load saved model and build the detection function
    detect_fn = tf2.saved_model.load(SAVED_MODEL_PATH)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info('Done! Took {} seconds'.format(elapsed_time))
    return detect_fn


def plot_bounding_box(image, bbox, label, color):
    (x_min, y_min, x_max, y_max) = bbox
    # bound box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 4, cv2.LINE_AA)
    # background for title
    text_offset_x = 5
    text_offset_y = 2
    (text_width, text_height) = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_PLAIN, fontScale=1., thickness=1)[0]
    cv2.rectangle(image, (x_min, y_min-text_height-2*text_offset_y), (x_min+text_width+text_offset_x *
                  2, y_min), color, cv2.FILLED)
    cv2.putText(image, label, (x_min+text_offset_x, y_min+text_offset_y),
                cv2.FONT_HERSHEY_PLAIN, 1., [0, 0, 0], 1, cv2.LINE_AA)


def filter_detections(detections, dims, threshold=0.75, show_spreader=False, max_detections=100):
    (width, height) = dims
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(
        np.int64)

    def criterio(filtered, h_margin=0.05, v_margin=0.00, min_area_overlap=.75):
        if filtered['class'] >= len(CLASSES.keys()):
            # we detected a class that we were not expecting, so we delete this detection
            return False
        if filtered['class'] != 1:
            if filtered['class'] == 2 and show_spreader:
                return True
            # Only apply criterion to containers
            return False
        area_bbox = (filtered['bbox'][2]-filtered['bbox']
                     [0])*(filtered['bbox'][3]-filtered['bbox'][1])
        # calculamos coordenadas del rectángulo que define el área de interés dentro de la imagen
        x0 = int(width*h_margin)
        xf = int((1-h_margin)*width)
        y0 = int(height*v_margin)
        yf = int(height*(1-v_margin))
        # area de la bbox dentro de la región de interés
        area = (min(xf, filtered['bbox'][2])-max(x0, filtered['bbox'][0])) * \
            (min(yf, filtered['bbox'][3])-max(y0, filtered['bbox'][1]))
        return area/area_bbox >= min_area_overlap

    filtered = []
    for i in range(0, min(num_detections, max_detections)):
        if detections['detection_scores'][i] >= threshold:
            detection = {}
            score = detections['detection_scores'][i]
            detection['score'] = score
            [y_min, x_min, y_max, x_max] = detections['detection_boxes'][i]
            detection['bbox'] = (int(x_min*width), int(y_min*height),
                                 int(x_max*width), int(y_max*height))
            class_type = detections['detection_classes'][i]
            detection['class'] = class_type
            detection['title'] = f"{CLASSES[str(class_type)]['label']}: {int(score*100)}%"
            detection['color'] = CLASSES[str(class_type)]['color']
            detection['focused_color'] = CLASSES[str(
                class_type)]['focused_color']
            filtered.append(detection)
    return [f for f in filtered if criterio(f)]


def stream_writer_rtsp_stream(width, height, fps, name, host, crane='rtg01', port=8554):
    ffmpeg_cmd = [
        "ffmpeg",
        "-f", "rawvideo",
        "-pixel_format", "bgr24",
        "-video_size", f"{width}x{height}",
        "-framerate", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "zerolatency",
        # Salida en formato rtsp
        "-f", "rtsp",
        "-rtsp_flags", "listen",
        f"rtsp://{host}:{port}/{crane}/{name}"
    ]
    return subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def stream_writer_dash_stream(width, height, fps, name,  crane='rtg01'):
    output_dir = os.path.join('/','var','media','hls', crane, name)
    os.makedirs(output_dir, exist_ok=True)
    ffmpeg_cmd = [
        "ffmpeg",
        "-f", "rawvideo",
        "-pixel_format", "bgr24",
        "-video_size", f"{width}x{height}",
        "-framerate", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        '-preset:v', 'veryfast',
        '-tune:v', 'zerolatency',
        "-f", "dash",
        "-seg_duration", '10',
        f"{output_dir}/stream.mpd"

    ]
    return subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

def stream_writer_hls_stream(width, height, fps, name,  crane='rtg01'):
    output_dir = os.path.join('/','var','media','hls', crane, name)
    os.makedirs(output_dir, exist_ok=True)
    ffmpeg_cmd = [
        "ffmpeg",
        "-f", "rawvideo",
        "-pixel_format", "bgr24",
        "-video_size", f"{width}x{height}",
        "-framerate", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        '-hls_time', '3',  # Duración de los segmentos HLS (en segundos)
        '-hls_list_size', '60',  # Tamaño de la lista de reproducción HLS
        '-hls_wrap', '100',  # Número máximo de archivos HLS antes de comenzar a sobrescribir
        '-start_number', '1',  # Número de inicio para los archivos HLS
        '-f', 'hls',  # Formato de salida HLS
        # Directorio de salida y nombre del archivo de lista de reproducción HLS
        f'{output_dir}/output.m3u8'
    ]
    return subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)


def stream_writer_rtp_process(width, height, fps):
    ffmpeg_cmd = [
        "ffmpeg",
        "-f", "rawvideo",
        "-pixel_format", "bgr24",
        "-video_size", f"{width}x{height}",
        "-framerate", str(fps),
        "-i", "-",
        "-c:v", "libx264",  # Puedes especificar el códec aquí
        "-preset", "fast",   # Opciones de rendimiento
        "-tune", "zerolatency",  # Opciones de latencia
        "-b:v", "900k",          # Bitrate de video
        "-bufsize", "900k",      # Tamaño del búfer
        "-maxrate", "900k",      # Tasa máxima de bits
        "-muxrate", "900k",      # Tasa de bits de multiplexación
        "-payload_type", "99",   # Tipo de carga útil RTP para H.264
        "-f", "rtp",             # Formato de salida RTP
        "-pkt_size", "1316",     # Tamaño máximo de paquete RTP
        "http://127.0.0.1:8554/stream"  # URL de salida HTTP
    ]
    return subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def main(args):
    hcrop = args['hcrop']
    vcrop = args['vcrop']
    max_detections = args['maxboxes']
    detection_threshold = max(0, min(1, args['detection_threshold']))
    # Origen del streaming de video de la cámara
    stream_url = args['input']  # r"./samples/video1.mp4"
    sync_fps = args['sync_fps']
    crane_name = args['crane_name']
    media_server_host = args['media_server_host']
    media_server_rtsp_port = args['media_server_rtsp_port']
    loop_video = args['loop']
    scheme = args['scheme']

    model = load_model()
    logger.info(f'Connected to video source "{stream_url}"...')
    cap = cv2.VideoCapture(stream_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        logger.error(f'Error opening video source at "{stream_url}"')
        return 1

    # Variable indicating if the spreader should be labeled with a box
    show_spreader = args['spreader']
    # Input frames will be scaled by this factor
    scale_factor = 1.

    # Output type: 0=no output, 1=labeled, 2=unlabelled, 3=both
    output_type = args['output_type']

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    target_width = int((width-2*hcrop)*scale_factor)
    target_height = int((height-2*vcrop)*scale_factor)
    logger.info(
        f"Receiving video from {stream_url} at {width}x{height} with {fps} fps")

    if scheme == 'hls':
        ffmpeg_process_single = stream_writer_hls_stream(
            target_width, target_height, fps, 'single', crane=crane_name)
        ffmpeg_process_dual = stream_writer_hls_stream(
            2*target_width, target_height, fps, 'dual', crane=crane_name)
    elif scheme == 'dash':
        ffmpeg_process_single = stream_writer_dash_stream(
            target_width, target_height, fps, 'single', crane=crane_name)
        ffmpeg_process_dual = stream_writer_dash_stream(
            2*target_width, target_height, fps, 'dual', crane=crane_name)
    else:
        # Iniciar el proceso ffmpeg
        ffmpeg_process_single = stream_writer_rtsp_stream(
            target_width, target_height, fps, 'single', crane=crane_name, host=media_server_host, port=media_server_rtsp_port)
        ffmpeg_process_dual = stream_writer_rtsp_stream(
            2*target_width, target_height, fps, 'dual', crane=crane_name, host=media_server_host, port=media_server_rtsp_port)
    frame_count = 0
    frames_to_skip = max(
        1, int(fps * max(0, min(args['skip_percentage']/100., 1.))))

    detections = []

    while True:
        if loop_video and not cap.isOpened():
            cap.open(stream_url)  # loop video when it finishes

        grabbed, input_frame = cap.read()
        start_time = time.time()
        if not grabbed:
            break
        # eliminamos pixeles que no son de interés
        input_frame = cv2.resize(
            input_frame[vcrop:height-vcrop, hcrop:width-hcrop], (target_width, target_height))
        frame_count = frame_count+1
        if frame_count % frames_to_skip == 0:
            # process alternate frames to downsample video
            scaled_frame = cv2.resize(input_frame, (640, 640))
            tensor = tf2.convert_to_tensor(scaled_frame[tf2.newaxis, ...])
            t0 = time.time()
            detections = model(tensor)
            t1 = time.time()
            detections = filter_detections(detections, (target_width, target_height),
                                           show_spreader=show_spreader, max_detections=max_detections, threshold=detection_threshold)
            logger.debug(f"Inference for frame {frame_count} done in {t1-t0}s")

        out_frame = cv2.resize(
            input_frame.copy(), (target_width, target_height))
        for detection in detections:
            plot_bounding_box(
                out_frame, detection['bbox'], detection['title'], detection['color'])

        if output_type == 0:
            cv2.imshow('RTG', out_frame)
        elif output_type == 1:
            cv2.imshow('RTG', input_frame)
        elif output_type == 2:
            cv2.imshow('RTG', np.concatenate((input_frame, out_frame), axis=1))
        try:
            ffmpeg_process_single.stdin.write(
                out_frame.astype(np.uint8).tobytes())
            ffmpeg_process_single.stdin.flush()
            ffmpeg_process_dual.stdin.write(np.concatenate(
                (input_frame, out_frame), axis=1).astype(np.uint8).tobytes())
            ffmpeg_process_dual.stdin.flush()
        except Exception as e:
            logger.error(e)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # exit app
            break
        elif key == ord('s'):
            # toggle spreader box visibility
            show_spreader = not show_spreader
        elif key == ord('t'):
            # change output type
            output_type = (output_type+1) % 3

        wait_time = 1/fps - (time.time()-start_time)
        if sync_fps and wait_time > 0:
            time.sleep(wait_time)
    cap.release()
    ffmpeg_process_single.stdin.close()
    ffmpeg_process_dual.stdin.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True,
                    help="Url of streaming or path to video file", type=str)
    ap.add_argument('-vcrop', required=False,
                    help='Vertical crop of input image, both at top and bottom', type=int, default=0)
    ap.add_argument('-hcrop', required=False,
                    help='Horizontal crop of input image, both at left and right sides', type=int, default=0)
    ap.add_argument('-maxboxes', required=False,
                    help='Max number of boxes to draw on the video', type=int, default=25)
    ap.add_argument('--sync_fps', type=bool, default=False, required=False,
                    help='If true, it will limit the input video fps to avoid reading frames faster than needed. This is useful when reading from a file instead of a net stream')
    ap.add_argument('--detection_threshold', required=False, type=float, default=0.75,
                    help="A number between 0 and 1 specifyng the detection threshold to displaye boxes. Defaults to 0.75")
    ap.add_argument('--spreader', required=False,
                    help='When True, the spreader box is displayed on the output video', type=bool, default=False)
    ap.add_argument('-t', '--output_type', help='Type of output video. When 0, video with labels is displayed; when 1, input video without labels is displayed; when 2, both videos are displayed; a different value disables output video', type=int, default=-1)
    ap.add_argument('-s', '--skip_percentage', type=int, default=50,
                    help="Percentage of frames to skip from the inference process.")
    ap.add_argument('--crane_name', type=str, default='rtg01',
                    required=False, help='The name of the crane')
    ap.add_argument('--media_server_host', required=True, type=str,
                    help='Host name of the mediser server receiving streams')
    ap.add_argument('--media_server_rtsp_port', required=False, type=int,
                    default=8554, help='Port associated to the RTSP protocol in the mediaserver')
    ap.add_argument('--media_server_hls_port', required=False, type=int,
                    default=8888, help='Port associated to the HLS protocol in the mediaserver')
    ap.add_argument('--loop', type=bool, required=False, default=False,
                    help='If true, the video will be looped indefinitely')
    ap.add_argument('--scheme', required=False, default='rtsp',
                    help='Scheme to use to publish streams to the media server')
    args = vars(ap.parse_args())
    main(args)
