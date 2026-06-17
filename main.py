import os
import cv2
import pywt
import pickle
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "data"
OUTPUT_DIR = "output"
WAVELET = "haar"
DWT_LEVEL = 4
SEED = 1234
IMAGE_SIZE = 512
WATERMARK_SIZE = 32
WATERMARK_NAME = "watermark.png"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_host_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"تصویر اصلی پیدا نشد: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    return image.astype(np.uint8)


def read_watermark(path):
    watermark = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if watermark is None:
        raise FileNotFoundError(f"واترمارک پیدا نشد: {path}")
    watermark = cv2.resize(watermark, (WATERMARK_SIZE, WATERMARK_SIZE))
    _, watermark = cv2.threshold(watermark, 127, 1, cv2.THRESH_BINARY)
    return watermark.astype(np.uint8)


def cpsnr(original, watermarked):
    original = original.astype(np.float64)
    watermarked = watermarked.astype(np.float64)
    mse = np.mean((original - watermarked) ** 2)
    if mse == 0:
        return 100.0
    return 10 * np.log10((255 ** 2) / mse)


def ber(original_wm, recovered_wm):
    return np.sum(original_wm != recovered_wm) / original_wm.size


def shuffle_bits(bits, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(bits))
    rng.shuffle(indices)
    return bits[indices], indices


def unshuffle_bits(bits, indices):
    result = np.zeros_like(bits)
    result[indices] = bits
    return result


def dwt4(channel):
    return pywt.wavedec2(channel, WAVELET, level=DWT_LEVEL)


def idwt4(coeffs):
    reconstructed = pywt.waverec2(coeffs, WAVELET)
    return np.clip(reconstructed, 0, 255)


def get_lh_hl(coeffs):
    cH, cV, cD = coeffs[1]
    HL = cH.copy()
    LH = cV.copy()
    return LH, HL


def set_lh_hl(coeffs, LH, HL):
    coeffs = list(coeffs)
    cH, cV, cD = coeffs[1]
    coeffs[1] = (HL, LH, cD)
    return coeffs


def otsu_threshold(values):
    values = np.array(values, dtype=np.float64)
    mean_value = np.mean(values)
    values = np.where(values > mean_value, mean_value, values)
    min_value = values.min()
    max_value = values.max()
    if max_value == min_value:
        return mean_value
    normalized = ((values - min_value) / (max_value - min_value) * 255).astype(np.uint8)
    histogram = np.bincount(normalized, minlength=256)
    total = normalized.size
    sum_total = np.dot(np.arange(256), histogram)
    sum_background = 0
    weight_background = 0
    max_variance = 0
    threshold = 0
    for t in range(256):
        weight_background += histogram[t]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += t * histogram[t]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance_between > max_variance:
            max_variance = variance_between
            threshold = t
    return min_value + (threshold / 255) * (max_value - min_value)


def embed_watermark(image, watermark):
    image_float = image.astype(np.float32)
    watermark_bits = watermark.flatten()
    shuffled_bits, shuffle_key = shuffle_bits(watermark_bits, SEED)
    coeffs_list = []
    lh_list = []
    hl_list = []
    diff_list = []
    for channel_index in range(3):
        coeffs = dwt4(image_float[:, :, channel_index])
        LH, HL = get_lh_hl(coeffs)
        coeffs_list.append(coeffs)
        lh_list.append(LH)
        hl_list.append(HL)
        diff_list.append(np.abs(LH - HL).flatten())
    diff_array = np.array(diff_list)
    number_of_bits = len(shuffled_bits)
    all_diffs = diff_array[:, :number_of_bits].flatten()
    delta1 = np.percentile(all_diffs, 30)
    delta2 = np.percentile(all_diffs, 75)
    if delta2 <= delta1:
        delta2 = delta1 + 20
    selected_channels = []
    selected_positions = []
    for bit_index, bit in enumerate(shuffled_bits):
        target = delta1 if bit == 0 else delta2
        diffs_for_bit = diff_array[:, bit_index]
        best_channel = int(np.argmin(np.abs(diffs_for_bit - target)))
        selected_channels.append(best_channel)
        selected_positions.append(bit_index)
        LH = lh_list[best_channel]
        HL = hl_list[best_channel]
        rows, cols = LH.shape
        row = bit_index // cols
        col = bit_index % cols
        if row >= rows:
            continue
        current_diff = abs(LH[row, col] - HL[row, col])
        if bit == 0:
            change = delta1 - current_diff
            if LH[row, col] >= HL[row, col]:
                LH[row, col] += change
            else:
                HL[row, col] += change
        else:
            if current_diff < delta2:
                change = (delta2 - current_diff) / 2
                if LH[row, col] >= HL[row, col]:
                    LH[row, col] += change
                    HL[row, col] -= change
                else:
                    LH[row, col] -= change
                    HL[row, col] += change
        lh_list[best_channel] = LH
        hl_list[best_channel] = HL
    watermarked_channels = []
    for channel_index in range(3):
        new_coeffs = set_lh_hl(coeffs_list[channel_index], lh_list[channel_index], hl_list[channel_index])
        reconstructed = idwt4(new_coeffs)
        watermarked_channels.append(reconstructed)
    watermarked = np.stack(watermarked_channels, axis=2)
    watermarked = np.clip(watermarked, 0, 255).astype(np.uint8)
    key = {
        "shuffle_key": shuffle_key,
        "selected_channels": selected_channels,
        "selected_positions": selected_positions,
        "watermark_shape": watermark.shape,
        "delta1": delta1,
        "delta2": delta2
    }
    return watermarked, key


def extract_watermark(watermarked, key):
    image_float = watermarked.astype(np.float32)
    diff_list = []
    for channel_index in range(3):
        coeffs = dwt4(image_float[:, :, channel_index])
        LH, HL = get_lh_hl(coeffs)
        diff_list.append(np.abs(LH - HL).flatten())
    extracted_diffs = []
    for channel_index, position in zip(key["selected_channels"], key["selected_positions"]):
        extracted_diffs.append(diff_list[channel_index][position])
    threshold = otsu_threshold(extracted_diffs)
    extracted_bits = np.array([1 if value >= threshold else 0 for value in extracted_diffs], dtype=np.uint8)
    recovered_bits = unshuffle_bits(extracted_bits, key["shuffle_key"])
    recovered_watermark = recovered_bits.reshape(key["watermark_shape"])
    return recovered_watermark, threshold


def show_results(original, watermarked, watermark, recovered):
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(2, 2, 2)
    plt.imshow(watermarked)
    plt.title("Watermarked Image")
    plt.axis("off")
    plt.subplot(2, 2, 3)
    plt.imshow(watermark, cmap="gray")
    plt.title("Original Watermark")
    plt.axis("off")
    plt.subplot(2, 2, 4)
    plt.imshow(recovered, cmap="gray")
    plt.title("Recovered Watermark")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "result_preview.png"), dpi=200)
    plt.show()


def run_single_image(host_image_name):
    host_path = os.path.join(DATA_DIR, host_image_name)
    watermark_path = os.path.join(DATA_DIR, WATERMARK_NAME)
    original = read_host_image(host_path)
    watermark = read_watermark(watermark_path)
    watermarked, key = embed_watermark(original, watermark)
    recovered, threshold = extract_watermark(watermarked, key)
    base_name = os.path.splitext(host_image_name)[0]
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_original.png"), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_watermarked.png"), cv2.cvtColor(watermarked, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_original_watermark.png"), watermark * 255)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_recovered_watermark.png"), recovered * 255)
    with open(os.path.join(OUTPUT_DIR, f"{base_name}_key.pkl"), "wb") as file:
        pickle.dump(key, file)
    return {
        "image": host_image_name,
        "cpsnr": cpsnr(original, watermarked),
        "ber": ber(watermark, recovered),
        "threshold": threshold,
        "original": original,
        "watermarked": watermarked,
        "watermark": watermark,
        "recovered": recovered
    }


def save_summary(results):
    summary_path = os.path.join(OUTPUT_DIR, "results_summary.csv")
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("Image,CPSNR,BER,OtsuThreshold\n")
        for result in results:
            file.write(f"{result['image']},{result['cpsnr']:.4f},{result['ber']:.6f},{result['threshold']:.4f}\n")


def main():
    image_names = ["lena.png", "peppers.png", "house.png", "airplane.png"]
    watermark_path = os.path.join(DATA_DIR, WATERMARK_NAME)
    if not os.path.exists(watermark_path):
        raise FileNotFoundError("فایل data/watermark.png پیدا نشد. اگر اسم فایل watermark.png.png است، آن را به watermark.png تغییر بده.")
    existing_images = [name for name in image_names if os.path.exists(os.path.join(DATA_DIR, name))]
    if not existing_images:
        raise FileNotFoundError("هیچ تصویر معتبری داخل پوشه data پیدا نشد.")
    results = []
    for image_name in existing_images:
        result = run_single_image(image_name)
        results.append(result)
        print("====================================")
        print(f"Image: {result['image']}")
        print(f"CPSNR = {result['cpsnr']:.2f} dB")
        print(f"BER = {result['ber']:.4f}")
        print(f"Otsu Threshold = {result['threshold']:.4f}")
        print("====================================")
    save_summary(results)
    first = results[0]
    show_results(first["original"], first["watermarked"], first["watermark"], first["recovered"])
    print("فایل‌های خروجی داخل پوشه output ذخیره شدند.")


if __name__ == "__main__":
    main()
