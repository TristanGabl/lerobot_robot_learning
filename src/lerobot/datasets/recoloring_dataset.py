import cv2
import numpy as np
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset
import json
import bisect

def hsv_recolor_preserve_shading(
    frame_bgr: np.ndarray,
    mask_u8: np.ndarray,
    *,
    target_hue: int,
    target_sat: int,
    value_factor: float = 1.0,
    alpha_blur: int = 9,
    mask_threshold: int = 127,
) -> np.ndarray:    
    mask = mask_u8 > mask_threshold
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    recolored_hsv = hsv.copy()
    recolored_hsv[..., 0][mask] = np.clip(target_hue, 0, 179)
    recolored_hsv[..., 1][mask] = np.clip(target_sat, 0, 255)

    if value_factor != 1.0:
        v = recolored_hsv[..., 2].astype(np.float32)
        v[mask] *= value_factor
        recolored_hsv[..., 2] = np.clip(v, 0, 255).astype(np.uint8)

    recolored_bgr = cv2.cvtColor(recolored_hsv, cv2.COLOR_HSV2BGR)
    alpha = mask.astype(np.float32) * 255.0
    if alpha_blur > 0:
        if alpha_blur % 2 == 0:
            alpha_blur += 1
        alpha = cv2.GaussianBlur(alpha, (alpha_blur, alpha_blur), 0)

    alpha = (alpha / 255.0)[..., None]
    out = recolored_bgr.astype(np.float32) * alpha + frame_bgr.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


class RecoloringLeRobotDataset(Dataset):
    def __init__(self, 
                 dataset, 
                 masks_dir: str, 
                 dataset_root: str | None = None,
                 mask_manifest_path: str | None = None,
                 recolor_prob=1.0, 
                 target_hue_range=None, 
                 target_sat_range=None, 
                 value_factor_range=None, 
                 debug_dir=None):
        
        self.dataset = dataset
        self.masks_dir = Path(masks_dir)
        self.dataset_root = Path(dataset_root) if dataset_root else self.masks_dir.parent

        manifest_path = Path(mask_manifest_path) if mask_manifest_path else self.masks_dir / "manifest.json"
        self.mask_manifest = json.loads(manifest_path.read_text())

        self._mask_starts = {}
        for cam, entries in self.mask_manifest.items():
            entries = sorted(entries, key=lambda e: e["global_start"])
            self.mask_manifest[cam] = entries
            self._mask_starts[cam] = [int(e["global_start"]) for e in entries]

        self.recolor_prob = recolor_prob
        self.target_hue_range = target_hue_range or [0, 179]
        self.target_sat_range = target_sat_range or [100, 255]
        self.value_factor_range = value_factor_range or [0.8, 1.2]
        self.debug_dir = Path(debug_dir) if debug_dir else None
        
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        # EXTRACT DATASET TRANSFORMS AND DISABLE THEM INTERNALLY
        self.image_transforms = self.dataset.image_transforms
        self.dataset.set_image_transforms(None) # NOTE: we will apply these manually after recoloring to ensure correct order of operations
        
        # Keep meta reference to act as a proper wrapper
        self.meta = getattr(self.dataset, "meta", None)
        self.camera_keys = self.meta.camera_keys if self.meta else []

    
    def _resolve_mask_path(self, cam: str, global_idx: int) -> Path | None:
        entries = self.mask_manifest.get(cam)
        if not entries:
            return None

        starts = self._mask_starts.get(cam, [])
        if not starts:
            return None

        pos = bisect.bisect_right(starts, global_idx) - 1

        if pos < 0:
            return None

        entry = entries[pos]

        if not (entry["global_start"] <= global_idx < entry["global_end_exclusive"]):
            return None

        local_idx = global_idx - int(entry["global_start"])
        return self.dataset_root / entry["mask_dir"] / f"{local_idx:06d}.png"
    
    def __getattr__(self, name: str):
        if name in ['dataset', 'masks_dir', 'recolor_prob', 'target_hue_range', 'target_sat_range', 'value_factor_range', 'debug_dir', 'image_transforms', 'meta', 'camera_keys']:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get RAW Untransformed item
        item = self.dataset[idx]
        
        abs_idx = item['index'].item() if isinstance(item['index'], torch.Tensor) else item['index']
        ep_idx = item['episode_index'].item() if isinstance(item['episode_index'], torch.Tensor) else item['episode_index']
        
        ep_start = self.meta.episodes["dataset_from_index"][ep_idx]
        ep_end = self.meta.episodes["dataset_to_index"][ep_idx]
        
        for cam in self.camera_keys:
            if cam not in item:
                continue
                
            # Randomly decide whether to skip recoloring for this camera's temporal stack entirely
            if torch.rand(1).item() > self.recolor_prob:
                if self.image_transforms is not None:
                    item[cam] = self.image_transforms(item[cam])
                continue

            # Delta temporal calculations
            if hasattr(self.dataset, 'delta_indices') and cam in self.dataset.delta_indices:
                deltas = self.dataset.delta_indices[cam]
            elif getattr(self.dataset, 'reader', None) and hasattr(self.dataset.reader, 'delta_indices') and cam in self.dataset.reader.delta_indices:
                deltas = self.dataset.reader.delta_indices[cam]
            else:
                deltas = [0]
                
            clamped_abs_indices = [max(ep_start, min(ep_end - 1, abs_idx + d)) for d in deltas]
            frame_indices_for_T = [abs_i - ep_start for abs_i in clamped_abs_indices]
            
            # Extract PyTorch Tensor (likely Float32 [0.0, 1.0])
            im_stack = item[cam] 
            is_temporal = (len(im_stack.shape) == 4)
            if not is_temporal:
                im_stack = im_stack.unsqueeze(0)
            
            _, _, out_H, out_W = im_stack.shape
            recolored_frames = []
            
            # Roll random properties identically for the entire temporal stack T
            sampled_hue = int(torch.randint(self.target_hue_range[0], self.target_hue_range[1] + 1, (1,)).item())
            sampled_sat = int(torch.randint(self.target_sat_range[0], self.target_sat_range[1] + 1, (1,)).item())
            sampled_val = (torch.rand(1) * (self.value_factor_range[1] - self.value_factor_range[0]) + self.value_factor_range[0]).item()
            
            for t_step, frame_idx in enumerate(frame_indices_for_T):
                global_idx = int(clamped_abs_indices[t_step])
                #mask_path = self.masks_dir / cam / f"episode_{ep_idx:05d}" / f"{frame_idx:05d}.png"
                #mask_path = self.masks_dir /f"{global_idx:06d}.png"
                mask_path = self._resolve_mask_path(cam, global_idx)
                
                # Load & Resize Mask
                if mask_path is not None and mask_path.exists():
                    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask_img is not None:
                        if mask_img.shape[:2] != (out_H, out_W):
                            mask_img = cv2.resize(mask_img, (out_W, out_H), interpolation=cv2.INTER_NEAREST)
                    else:
                        mask_img = np.zeros((out_H, out_W), dtype=np.uint8)
                        print(f"Warning: Failed to load mask image at {mask_path}. Using empty mask.")
                else:
                    mask_img = np.zeros((out_H, out_W), dtype=np.uint8)
                    print(f"Warning: Mask image not found at {mask_path}. Using empty mask.")

                # Convert PyTorch RGB frame -> OpenCV BGR uint8
                frame_rgb = im_stack[t_step].numpy()
                if frame_rgb.dtype in (np.float32, np.float64):
                    frame_rgb_u8 = (np.clip(frame_rgb, 0, 1) * 255).astype(np.uint8)
                else:
                    frame_rgb_u8 = frame_rgb
                    
                frame_bgr_u8 = cv2.cvtColor(np.transpose(frame_rgb_u8, (1, 2, 0)), cv2.COLOR_RGB2BGR)
                
                # Apply script execution exactly using consistent randomized params
                recolored_bgr = hsv_recolor_preserve_shading(
                    frame_bgr=frame_bgr_u8,
                    mask_u8=mask_img,
                    target_hue=sampled_hue,
                    target_sat=sampled_sat,
                    value_factor=sampled_val
                )

                # Optionally debug save visually-verifiable samples
                if self.debug_dir and torch.rand(1).item() < 0.05:
                    safe_cam = cam.replace(".", "_").replace("/", "_")

                    save_path = self.debug_dir / (
                        f"{safe_cam}_ep{ep_idx:03d}_global{global_idx:06d}_local{frame_idx:05d}_"
                        f"h{sampled_hue}_s{sampled_sat}.jpg"
                    )

                    mask_save_path = self.debug_dir / (
                        f"{safe_cam}_ep{ep_idx:03d}_global{global_idx:06d}_local{frame_idx:05d}_mask.png"
                    )

                    if not save_path.exists():
                        print(f"Saving debug recolored image to: {save_path}")
                        cv2.imwrite(str(save_path), recolored_bgr)
                        cv2.imwrite(str(mask_save_path), mask_img)
                
                # Convert back -> PyTorch RGB float
                #recolored_rgb = cv2.cvtColor(recolored_bgr, cv2.COLOR_BGR2RGB)

                # ----------PATCH
                orig_dtype = im_stack.dtype

                recolored_rgb = cv2.cvtColor(recolored_bgr, cv2.COLOR_BGR2RGB)
                recolored_chw_u8 = np.transpose(recolored_rgb, (2, 0, 1))
                #recolored_tensor = torch.from_numpy(np.transpose(recolored_rgb, (2, 0, 1))).float() / 255.0
                recolored_tensor = torch.from_numpy(recolored_chw_u8)

                if orig_dtype.is_floating_point:
                    recolored_tensor = recolored_tensor.float() / 255.0
                else:
                    recolored_tensor = recolored_tensor.to(orig_dtype)
                                
                recolored_frames.append(recolored_tensor)
                # ----------------------------------
            
            # Reconstruct Tensor Stack
            recolored_stack = torch.stack(recolored_frames)
            if not is_temporal:
                recolored_stack = recolored_stack.squeeze(0)
                
            item[cam] = recolored_stack

            # 3. APPLY STORED NORMAL DATASET TRANSFORMS LAST (e.g. ColorJitter)            
            if self.image_transforms is not None:
                item[cam] = self.image_transforms(item[cam])

            if self.debug_dir and torch.rand(1).item() < 0.05:
                safe_cam = cam.replace(".", "_").replace("/", "_")
                out = self.debug_dir / (
                    f"{safe_cam}_ep{ep_idx:03d}_global{abs_idx:06d}_TEMPORAL_AFTER.jpg"
                )
                print(f"DEBUG {cam} tensor shape after transforms:", item[cam].shape)
                print("camera", cam, "deltas", deltas)
                print("num temporal frames:", len(deltas))
                self._save_temporal_stack_debug(item[cam], out, max_frames=32)
                print(
                    "AFTER debug:",
                    cam,
                    item[cam].dtype,
                    item[cam].min().item(),
                    item[cam].max().item(),
                    item[cam].shape,
                )

        # Extra check to assert the same LeRobot format
        for cam in self.camera_keys:
            if cam in item:
                assert item[cam].dtype == torch.uint8, (
                    cam, item[cam].dtype, item[cam].min().item(), item[cam].max().item()
                )      
        return item


    def _save_tensor_debug_image(
        self,
        tensor: torch.Tensor,
        path: Path,
    ) -> None:
        """
        Save a torch image tensor as a debug jpg/png.

        Accepts:
        [C, H, W]
        [T, C, H, W] -> saves first temporal frame
        Assumes RGB float tensor in [0, 1].
        """
        if tensor.ndim == 4:
            tensor = tensor[0]  # first temporal frame

        img = tensor.detach().cpu().numpy()

        # CHW -> HWC
        img = np.transpose(img, (1, 2, 0))

        # float [0,1] -> uint8 [0,255]
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        # RGB -> BGR for OpenCV
        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img_bgr)


    def _save_temporal_stack_debug(
        self,
        tensor: torch.Tensor,
        path: Path,
        max_frames: int = 16,
    ) -> None:
        """
        Save [T,C,H,W] or [C,H,W] tensor as a horizontal strip.
        Supports:
        float tensors in [0,1]
        uint8 tensors in [0,255]
        Assumes RGB channel order.
        """
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        frames = []
        for t in range(min(tensor.shape[0], max_frames)):
            frame = tensor[t].detach().cpu()

            # CHW -> HWC
            img = frame.permute(1, 2, 0).numpy()

            if img.dtype == np.uint8:
                img_u8 = img
            else:
                img_u8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)

            img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

            cv2.putText(
                img_bgr,
                f"t={t}",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            frames.append(img_bgr)

        strip = np.concatenate(frames, axis=1)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), strip)