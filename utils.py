import numpy as np
from pathlib import Path

def xywh2xyxy(box):
    """
    Convierte un bounding box en formato (x, y, w, h) centrado
    al formato (x1, y1, x2, y2) de esquinas.

    Parámetros
    ----------
    box : array-like
        Bounding box en formato YOLO (x_centro, y_centro, ancho, alto).
        Puede ser 1D (una sola caja) o 2D (múltiples cajas).

    Retorna
    -------
    numpy.ndarray
        Bounding box convertido(s) a formato esquinas.
    """
    b = np.asarray(box, dtype=float)
    if b.ndim == 1:
        x, y, w, h = b
        return np.array([x - w/2, y - h/2, x + w/2, y + h/2])
    else:
        x, y, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        return np.stack([x - w/2, y - h/2, x + w/2, y + h/2], axis=1)


def box_iou(boxes1, boxes2):
    """
    Calcula la matriz IoU entre dos conjuntos de cajas.

    Parámetros
    ----------
    boxes1 : numpy.ndarray
        Cajas en formato xyxy de forma (N,4).
    boxes2 : numpy.ndarray
        Cajas en formato xyxy de forma (M,4).

    Retorna
    -------
    numpy.ndarray
        Matriz IoU de tamaño (N,M).
    """
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=float)

    b1 = boxes1[:, None, :]
    b2 = boxes2[None, :, :]

    inter_x1 = np.maximum(b1[..., 0], b2[..., 0])
    inter_y1 = np.maximum(b1[..., 1], b2[..., 1])
    inter_x2 = np.minimum(b1[..., 2], b2[..., 2])
    inter_y2 = np.minimum(b1[..., 3], b2[..., 3])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter_area = inter_w * inter_h

    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])

    union = area1 + area2 - inter_area + 1e-9
    return inter_area / union


def compute_map(pred_dir, gt_dir, iou_thresh=0.5):
    """
    Calcula mAP@IoU utilizando predicciones YOLO y ground truth
    guardados como archivos .txt en carpetas separadas.

    Parámetros
    ----------
    pred_dir : str o Path
        Carpeta con predicciones YOLO.
    gt_dir : str o Path
        Carpeta con ground truths YOLO.
    iou_thresh : float
        Umbral IoU para considerar una predicción como verdadera positiva.

    Retorna
    -------
    mAP : float
        Mean Average Precision (promedio sobre clases).
    aps : dict
        AP por clase {clase: AP}.
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)

    gt_boxes = {}
    all_classes = set()

    for gt_file in gt_dir.glob("*.txt"):
        image_id = gt_file.stem
        lines = gt_file.read_text().strip().splitlines()
        boxes = []
        for ln in lines:
            parts = ln.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            box = np.array([float(v) for v in parts[1:5]], dtype=float)
            bbox = xywh2xyxy(box)
            boxes.append({"cls": cls, "bbox": bbox, "used": False})
            all_classes.add(cls)
        gt_boxes[image_id] = boxes

    preds = []
    for pred_file in pred_dir.glob("*.txt"):
        image_id = pred_file.stem
        lines = pred_file.read_text().strip().splitlines()
        for ln in lines:
            parts = ln.strip().split()
            if len(parts) < 6:
                continue
            cls = int(parts[0])
            box = np.array([float(v) for v in parts[1:5]], dtype=float)
            conf = float(parts[5])
            bbox = xywh2xyxy(box)
            preds.append({"image_id": image_id, "cls": cls, "bbox": bbox, "conf": conf})
            all_classes.add(cls)

    if not preds:
        print("No se encontraron predicciones en", pred_dir)
        return 0.0, {}

    aps = {}
    all_classes = sorted(list(all_classes))

    for cls in all_classes:
        cls_preds = [p for p in preds if p["cls"] == cls]

        npos = 0
        for image_id, boxes in gt_boxes.items():
            npos += sum(1 for b in boxes if b["cls"] == cls)

        if npos == 0:
            continue

        cls_preds.sort(key=lambda x: x["conf"], reverse=True)

        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))

        for i, pred in enumerate(cls_preds):
            image_id = pred["image_id"]
            pred_box = pred["bbox"][None, :]

            gts_img = [b for b in gt_boxes.get(image_id, []) if b["cls"] == cls]
            if not gts_img:
                fp[i] = 1
                continue

            gt_boxes_np = np.stack([g["bbox"] for g in gts_img], axis=0)
            ious = box_iou(pred_box, gt_boxes_np)[0]

            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]

            if max_iou >= iou_thresh and not gts_img[max_iou_idx]["used"]:
                tp[i] = 1
                gts_img[max_iou_idx]["used"] = True
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        recall = tp_cum / (npos + 1e-9)
        precision = tp_cum / (tp_cum + fp_cum + 1e-9)

        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

        aps[cls] = ap

    if not aps:
        print("No se pudo calcular AP.")
        return 0.0, {}

    mAP = float(np.mean(list(aps.values())))
    return mAP, aps


def load_preds_from_dir(pred_dir: Path, image_stems, conf_min):
    """
    Carga predicciones YOLO desde una carpeta y devuelve tuplas (img_id, score, box_xywh).
    Maneja distintos formatos de predicción (5, 6 columnas, etc.).

    Parámetros
    ----------
    pred_dir : Path
        Carpeta con predicciones YOLO.
    image_stems : lista
        Lista de nombres de imágenes (sin extensión).
    conf_min : float
        Umbral de confianza mínima.

    Retorna
    -------
    list
        Lista de tuplas (img_id, confianza, bbox_xywh).
    """
    all_preds = []
    pred_dir = Path(pred_dir)

    for stem in image_stems:
        p = pred_dir / f"{stem}.txt"
        if not p.exists():
            continue

        arr = np.loadtxt(p, ndmin=2)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr[None, :]

        ncol = arr.shape[1]

        if ncol == 6:
            col_last = arr[:, 5]
            if np.all((col_last >= 0) & (col_last <= 1)):
                conf = col_last
                xywh = arr[:, 1:5]
            else:
                conf = arr[:, 1]
                xywh = arr[:, 2:6]

        elif ncol == 5:
            conf = np.full(arr.shape[0], 0.5, dtype=float)
            xywh = arr[:, 1:5]

        else:
            continue

        mask = conf >= conf_min
        conf = conf[mask]
        xywh = xywh[mask]

        for c, box in zip(conf, xywh):
            all_preds.append((stem, float(c), box.astype(float)))

    return all_preds


def compute_map50(gt, preds, iou_thr=0.5):
    """
    Calcula Precision, Recall y AP@0.5 para un conjunto de predicciones y GT ya cargados.

    Parámetros
    ----------
    gt : dict
        Diccionario con GT por imagen: {"boxes": array(N,4)}.
    preds : list
        Lista de predicciones (img_id, confianza, box_xywh).
    iou_thr : float
        Umbral IoU para match.

    Retorna
    -------
    tuple (P, R, AP50)
        Precisión final, Recall final, y Average Precision.
    """
    gt2 = {
        k: {
            "boxes": np.asarray(v["boxes"], dtype=float),
            "detected": np.zeros(len(v["boxes"]), dtype=bool),
        }
        for k, v in gt.items()
    }

    preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
    n = len(preds_sorted)
    tp = np.zeros(n)
    fp = np.zeros(n)
    total_gt = sum(len(v["boxes"]) for v in gt2.values())

    for i, (img, score, box) in enumerate(preds_sorted):
        if img not in gt2 or gt2[img]["boxes"].size == 0:
            fp[i] = 1
            continue
        boxes = gt2[img]["boxes"]
        ious = box_iou(box, boxes)
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr and not gt2[img]["detected"][j]:
            tp[i] = 1
            gt2[img]["detected"][j] = True
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / (total_gt + 1e-9)
    precision = cum_tp / (cum_tp + cum_fp + 1e-9)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

    P = float(precision[-1]) if precision.size > 0 else 0.0
    R = float(recall[-1]) if recall.size > 0 else 0.0
    return P, R, ap


def evaluar_modelo(nombre, pred_dir, conf_minn, gt_dict):
    """
    Evalúa un modelo YOLO a partir de un directorio de predicciones y GT cargado.

    Parámetros
    ----------
    nombre : str
        Nombre del modelo (para imprimir resultados).
    pred_dir : str o Path
        Carpeta con predicciones YOLO.
    conf_minn : float
        Umbral mínimo de confianza.
    gt_dict : dict
        Ground truth cargado con load_gt_dict().

    Imprime
    -------
    Precision, Recall, F1 y mAP@0.5.
    """
    preds = load_preds_from_dir(Path(pred_dir), conf_min=conf_minn)

    P, R, mAP50 = compute_map50(gt_dict, preds, iou_thr=0.5)
    F1 = 2 * P * R / (P + R + 1e-9)

    print(f"===== Resultados TEST – Modelo {nombre} =====")
    print(f"Predicciones totales: {len(preds)}")
    print(f"Precision: {P:.3f}")
    print(f"Recall:    {R:.3f}")
    print(f"F1-score:  {F1:.3f}")
    print(f"mAP@0.5:   {mAP50:.3f}")
    print()


def nms(preds, iou_thr=0.5):
    """
    Aplica Non-Maximum Suppression (NMS) a un conjunto de predicciones.

    Parámetros
    ----------
    preds : array (N,5)
        Formato: [x, y, w, h, score].
    iou_thr : float
        Umbral IoU para suprimir detecciones redundantes.

    Retorna
    -------
    numpy.ndarray
        Predicciones que sobreviven al NMS.
    """
    if len(preds) == 0:
        return preds

    preds = np.asarray(preds, dtype=float)
    scores = preds[:, 4]
    order = np.argsort(scores)[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou(preds[i, :4], preds[order[1:], :4])
        inds = np.where(ious < iou_thr)[0]
        order = order[inds + 1]

    return preds[keep]


def load_yolo_preds_txt(path, conf_min=0.0):
    """
    Carga un archivo .txt de predicciones YOLO y devuelve array (N,5)
    con columnas [x, y, w, h, conf], filtrando por confianza.

    Parámetros
    ----------
    path : Path o str
        Ruta del archivo de predicciones.
    conf_min : float
        Umbral mínimo de confianza.

    Retorna
    -------
    numpy.ndarray
        Predicciones filtradas.
    """
    arr = np.loadtxt(path, ndmin=2)
    if arr.size == 0:
        return np.zeros((0, 5), dtype=float)

    if arr.ndim == 1:
        arr = arr[None, :]

    ncol = arr.shape[1]
    if ncol == 6:
        xywh = arr[:, 1:5]
        conf = arr[:, 5]
    elif ncol == 5:
        xywh = arr[:, 1:5]
        conf = np.full(arr.shape[0], 0.5, dtype=float)
    elif ncol == 4:
        xywh = arr[:, 0:4]
        conf = np.full(arr.shape[0], 0.01, dtype=float)
    else:
        raise ValueError(f"Formato inesperado en {path}, ncol={ncol}")

    xywh_conf = np.concatenate([xywh, conf[:, None]], axis=1)

    mask = xywh_conf[:, 4] >= conf_min
    return xywh_conf[mask]


def load_gt_dict(image_stems, gt_root):
    """
    Carga ground truths YOLO desde un directorio, en formato xywh normalizado.

    Parámetros
    ----------
    image_stems : lista
        Lista de nombres de imágenes sin extensión.
    gt_root : Path
        Carpeta donde están los .txt del GT.

    Retorna
    -------
    dict
        gt_dict[img_id] = {"boxes": array(N,4)}.
    """
    gt_dict = {}

    for stem in image_stems:
        gt_path = gt_root / f"{stem}.txt"
        boxes = []
        if gt_path.exists():
            try:
                g = np.loadtxt(gt_path, ndmin=2)
                if g.size > 0:
                    if g.ndim == 1:
                        g = g[None, :]
                    boxes = g[:, 1:5]
            except Exception as e:
                print(f"[WARN] Problema leyendo {gt_path}: {e}")
                boxes = []

        gt_dict[stem] = {"boxes": np.asarray(boxes, dtype=float)}

    return gt_dict


def build_fused_preds(image_stems,
                      rgb_pred_dir,
                      therm_pred_dir,
                      conf_min_rgb=0.25,
                      conf_min_therm=0.25,
                      weight_rgb=0.5,
                      weight_therm=0.5,
                      iou_thr_nms=0.5):
    """
    Genera predicciones fusionadas (late fusion) combinando resultados de
    detecciones RGB y Thermal, ponderando sus puntajes y aplicando NMS.

    Parámetros
    ----------
    image_stems : lista
        Lista de nombres de imágenes (sin extensión).
    rgb_pred_dir : Path
        Carpeta con predicciones RGB.
    therm_pred_dir : Path
        Carpeta con predicciones térmicas.
    conf_min_rgb : float
        Umbral de confianza mínimo para predicciones RGB.
    conf_min_therm : float
        Umbral de confianza mínimo para predicciones térmicas.
    weight_rgb : float
        Peso asignado a las predicciones RGB.
    weight_therm : float
        Peso asignado a las predicciones térmicas.
    iou_thr_nms : float
        Umbral IoU para NMS final.

    Retorna
    -------
    list
        Lista de tuplas (img_id, score, bbox_xywh) fusionadas.
    """
    fused_preds = []

    for stem in image_stems:
        all_preds_xywh_score = []

        pred_path_rgb = rgb_pred_dir / f"{stem}.txt"
        if pred_path_rgb.exists():
            xywh_conf_rgb = load_yolo_preds_txt(pred_path_rgb, conf_min=conf_min_rgb)
            if xywh_conf_rgb.size > 0:
                scores_rgb = xywh_conf_rgb[:, 4] * weight_rgb
                all_preds_xywh_score.append(
                    np.concatenate([xywh_conf_rgb[:, :4], scores_rgb[:, None]], axis=1)
                )

        pred_path_therm = therm_pred_dir / f"{stem}.txt"
        if pred_path_therm.exists():
            xywh_conf_therm = load_yolo_preds_txt(pred_path_therm, conf_min=conf_min_therm)
            if xywh_conf_therm.size > 0:
                scores_therm = xywh_conf_therm[:, 4] * weight_therm
                all_preds_xywh_score.append(
                    np.concatenate([xywh_conf_therm[:, :4], scores_therm[:, None]], axis=1)
                )

        if all_preds_xywh_score:
            all_preds_xywh_score = np.concatenate(all_preds_xywh_score, axis=0)
            fused = nms(all_preds_xywh_score, iou_thr=iou_thr_nms)
            for row in fused:
                xywh = row[:4]
                score = float(row[4])
                fused_preds.append((stem, score, xywh))

    return fused_preds
