"""Shared spatial preprocessing for PIDNet training and evaluation."""

from __future__ import annotations

import argparse

import cv2


DEFAULT_IMAGE_SIZE = "1920x1080"


def parse_image_size(value):
    """Return ``(width, height)`` from a supported image-size specification."""

    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise argparse.ArgumentTypeError("Image size tuple must be (width, height).")
        width, height = value
    else:
        text = str(value).strip().lower()
        if text in {"1080p", "hd1080"}:
            return 1920, 1080
        if text in {"720p", "hd720"}:
            return 1280, 720
        text = text.replace(",", "x").replace("*", "x")
        if "x" in text:
            parts = [part.strip() for part in text.split("x") if part.strip()]
            if len(parts) != 2:
                raise argparse.ArgumentTypeError(
                    "Use WIDTHxHEIGHT, for example 1920x1080."
                )
            width, height = parts
        else:
            width = height = text
    try:
        width, height = int(width), int(height)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "Image size must be an integer or WIDTHxHEIGHT."
        ) from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Image width and height must be positive.")
    return width, height


def image_size_text(image_size):
    width, height = parse_image_size(image_size)
    return f"{width}x{height}" if width != height else str(width)


def resize_image_and_mask(image, mask, image_size):
    """Aspect-fit an image/mask pair using the canonical letterbox operation."""

    target_w, target_h = parse_image_size(image_size)
    height, width = image.shape[:2]
    scale = min(target_w / max(width, 1), target_h / max(height, 1))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    left = (target_w - new_w) // 2
    right = target_w - new_w - left
    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    mask = cv2.copyMakeBorder(
        mask,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    return image, mask
