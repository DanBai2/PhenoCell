import numpy as np
import cv2
from skimage import filters, exposure, restoration, morphology
from scipy import ndimage
import tifffile
from matplotlib import pyplot as plt
import os
from pathlib import Path
import logging

class CellPaintingPreprocessor:
    """
    Cell PaintingImageProcessing
    Image、、Processing
    """
    
    def __init__(self, output_dir=None, log_level=logging.INFO):
        self.output_dir = output_dir
        self.setup_logging(log_level)
        
    def setup_logging(self, log_level):
        """Setup logging"""
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_tiff_image(self, file_path):
        """LoadTIFFImage"""
        try:
            image = tifffile.imread(file_path)
            self.logger.info(f"SuccessLoadImage: {file_path}, Shape: {image.shape}, Type: {image.dtype}")
            return image
        except Exception as e:
            self.logger.error(f"LoadImageFailed {file_path}: {e}")
            return None
    
    def detect_outliers_quality_control(self, image, method='zscore', threshold=3):
        """
        Image（）
        """
        if method == 'zscore':

            z_scores = np.abs((image - np.mean(image)) / np.std(image))
            outlier_ratio = np.sum(z_scores > threshold) / image.size
            return outlier_ratio < 0.01
        
        elif method == 'iqr':

            Q1 = np.percentile(image, 25)
            Q3 = np.percentile(image, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_ratio = np.sum((image < lower_bound) | (image > upper_bound)) / image.size
            return outlier_ratio < 0.01
        
        return True
    
    def calculate_image_quality_metrics(self, image):
        """ImageMetric"""
        metrics = {}
        

        signal_mean = np.mean(image)
        noise_std = np.std(image)
        metrics['snr_estimate'] = signal_mean / noise_std if noise_std > 0 else 0
        

        metrics['contrast'] = np.std(image)
        

        metrics['dynamic_range'] = np.max(image) - np.min(image)
        

        gy, gx = np.gradient(image.astype(float))
        metrics['sharpness'] = np.mean(gx**2 + gy**2)
        

        metrics['uniformity'] = 1 / (1 + np.var(ndimage.uniform_filter(image, size=10)))
        
        return metrics
    
    def remove_background_illumination(self, image, method='tophat', kernel_size=15):
        """
        
        """
        if method == 'tophat':

            kernel = morphology.disk(kernel_size)
            background = morphology.opening(image, kernel)
            corrected = image - background
            
        elif method == 'rolling_ball':

            kernel = morphology.ball(kernel_size)
            background = morphology.opening(image, kernel)
            corrected = image - background
            
        elif method == 'gaussian':

            background = ndimage.gaussian_filter(image, sigma=kernel_size)
            corrected = image - background
            
        else:
            corrected = image
            

        corrected = np.maximum(corrected, 0)
        return corrected
    
    def denoise_image(self, image, method='gaussian', **kwargs):
        """
        Image
        """
        if method == 'gaussian':

            sigma = kwargs.get('sigma', 1.0)
            denoised = ndimage.gaussian_filter(image, sigma=sigma)
            
        elif method == 'median':

            size = kwargs.get('size', 3)
            denoised = ndimage.median_filter(image, size=size)
            
        elif method == 'bilateral':

            diameter = kwargs.get('diameter', 5)
            sigma_color = kwargs.get('sigma_color', 50)
            sigma_space = kwargs.get('sigma_space', 50)
            

            img_8bit = self.normalize_to_8bit(image)
            denoised = cv2.bilateralFilter(img_8bit, diameter, sigma_color, sigma_space)
            denoised = denoised.astype(np.float32)
            
        elif method == 'tv_chambolle':

            weight = kwargs.get('weight', 0.1)
            denoised = restoration.denoise_tv_chambolle(image, weight=weight)
            
        elif method == 'wavelet':

            denoised = restoration.denoise_wavelet(image, method='BayesShrink', mode='soft')
            
        else:
            denoised = image
            
        return denoised
    
    def enhance_contrast(self, image, method='histogram_equalization', **kwargs):
        """
        
        """
        if method == 'histogram_equalization':

            enhanced = exposure.equalize_hist(image)
            
        elif method == 'adaptive_equalization':

            clip_limit = kwargs.get('clip_limit', 0.03)
            kernel_size = kwargs.get('kernel_size', None)
            enhanced = exposure.equalize_adapthist(image, clip_limit=clip_limit, kernel_size=kernel_size)
            
        elif method == 'gamma_correction':

            gamma = kwargs.get('gamma', 1.2)
            enhanced = exposure.adjust_gamma(image, gamma=gamma)
            
        elif method == 'log_correction':

            enhanced = exposure.adjust_log(image, 1)
            
        elif method == 'percentile_stretching':

            lower = kwargs.get('lower_percentile', 1)
            upper = kwargs.get('upper_percentile', 99)
            p_low, p_high = np.percentile(image, (lower, upper))
            enhanced = exposure.rescale_intensity(image, in_range=(p_low, p_high))
            
        else:
            enhanced = image
            
        return enhanced
    
    def normalize_to_8bit(self, image):
        """Image8"""
        if image.dtype == np.uint8:
            return image
        
        image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        return (image_normalized * 255).astype(np.uint8)
    
    def normalize_to_float(self, image):
        """Image0-1"""
        if image.dtype == np.float32 or image.dtype == np.float64:
            return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        
        return image.astype(np.float32) / np.iinfo(image.dtype).max
    
    def detect_and_remove_artifacts(self, image, artifact_threshold=0.9):
        """
        
        """

        binary = image > (np.percentile(image, 95))
        

        labeled, num_features = ndimage.label(binary)
        

        sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
        

        artifact_mask = np.zeros_like(image, dtype=bool)
        for i, size in enumerate(sizes):
            if size > image.size * artifact_threshold / 100:
                artifact_mask[labeled == i + 1] = True
        

        if np.any(artifact_mask):
            self.logger.warning("Image，")
            repaired = image.copy()
            repaired[artifact_mask] = ndimage.median_filter(image, size=3)[artifact_mask]
            return repaired
        
        return image
    
    def full_preprocessing_pipeline(self, image, config=None):
        """
        Processing
        """
        if config is None:
            config = {
                'quality_control': True,
                'background_removal': True,
                'denoising': True,
                'contrast_enhancement': True,
                'artifact_removal': True,
                'normalization': True
            }
        
        original_image = image.copy()
        processed = image.copy()
        

        if config.get('quality_control', True):
            metrics = self.calculate_image_quality_metrics(processed)
            self.logger.info(f"ImageMetric: {metrics}")
            
            if not self.detect_outliers_quality_control(processed):
                self.logger.warning("ImageFailed，")
        

        if config.get('background_removal', True):
            processed = self.remove_background_illumination(processed, method='tophat')
        

        if config.get('artifact_removal', True):
            processed = self.detect_and_remove_artifacts(processed)
        

        if config.get('denoising', True):
            processed = self.denoise_image(processed, method='gaussian')
        

        if config.get('contrast_enhancement', True):
            processed = self.enhance_contrast(processed, method='histogram_equalization')
        

        if config.get('normalization', True):
            processed = self.normalize_to_float(processed)
        
        return processed
    
    def batch_process_directory(self, input_dir, output_dir=None, file_pattern="*.tif", config=None):
        """
        ProcessingDirectoryImage
        """
        if output_dir is None:
            output_dir = self.output_dir or input_dir + "_processed"
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        input_path = Path(input_dir)
        image_files = list(input_path.rglob(file_pattern))
        
        self.logger.info(f" {len(image_files)} ImageFile")
        
        results = []
        for file_path in image_files:
            self.logger.info(f"ProcessingImage: {file_path.name}")
            

            image = self.load_tiff_image(file_path)
            if image is None:
                continue
            

            processed = self.full_preprocessing_pipeline(image, config)
            

            output_path = Path(output_dir) / f"processed_{file_path.name}"
            tifffile.imwrite(output_path, processed)
            

            original_metrics = self.calculate_image_quality_metrics(image)
            processed_metrics = self.calculate_image_quality_metrics(processed)
            
            results.append({
                'file_name': file_path.name,
                'original_metrics': original_metrics,
                'processed_metrics': processed_metrics,
                'output_path': output_path
            })
            
            self.logger.info(f"Processing: {file_path.name}")
        
        return results
    
    def visualize_preprocessing_results(self, original, processed, save_path=None):
        """
        ProcessingResults
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        

        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        

        axes[0, 1].imshow(processed, cmap='gray')
        axes[0, 1].set_title('Processed Image')
        axes[0, 1].axis('off')
        

        axes[0, 2].hist(original.flatten(), bins=50, alpha=0.7, label='Original')
        axes[0, 2].hist(processed.flatten(), bins=50, alpha=0.7, label='Processed')
        axes[0, 2].set_title('Histogram Comparison')
        axes[0, 2].legend()
        

        diff = processed - self.normalize_to_float(original)
        axes[1, 0].imshow(diff, cmap='coolwarm')
        axes[1, 0].set_title('Difference (Processed - Original)')
        axes[1, 0].axis('off')
        

        orig_metrics = self.calculate_image_quality_metrics(original)
        proc_metrics = self.calculate_image_quality_metrics(processed)
        
        metrics_names = ['SNR', 'Contrast', 'Sharpness']
        orig_values = [orig_metrics['snr_estimate'], orig_metrics['contrast'], orig_metrics['sharpness']]
        proc_values = [proc_metrics['snr_estimate'], proc_metrics['contrast'], proc_metrics['sharpness']]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, orig_values, width, label='Original')
        axes[1, 1].bar(x + width/2, proc_values, width, label='Processed')
        axes[1, 1].set_title('Quality Metrics Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics_names)
        axes[1, 1].legend()
        

        improvement = [(proc - orig) / orig * 100 for orig, proc in zip(orig_values, proc_values)]
        axes[1, 2].bar(metrics_names, improvement, color=['green' if x > 0 else 'red' for x in improvement])
        axes[1, 2].set_title('Improvement (%)')
        axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"ResultsSave: {save_path}")
        
        # plt.show()



def main():

    preprocessor = CellPaintingPreprocessor(output_dir="./processed_images")
    

    config = {
        'quality_control': True,
        'background_removal': True,
        'denoising': True,
        'contrast_enhancement': True,
        'artifact_removal': True,
        'normalization': True
    }
    

    image_path = "path/to/your/data/ours/img/Control-Model-Positive-Yao/Batch1/Batch1-Plate1-20251017_B02_s2_w5A20E9901-A322-441D-A062-6EDAE81050A4.tif"
    image = preprocessor.load_tiff_image(image_path)
    processed = preprocessor.full_preprocessing_pipeline(image, config)
    preprocessor.visualize_preprocessing_results(image, processed, "path/to/your/project/results/img_cls/Batch1-Plate1-20251017_B02_s2_w5A20E9901_preprocessing_result.png")
    

    # input_directory = "path/to/your/data/ours/img/Control-Model-Positive-Yao/Batch1/Batch1-Plate1-Merge"
    # results = preprocessor.batch_process_directory(input_directory, config=config)
    


    # snr_improvements = []
    # contrast_improvements = []
    
    # for result in results:
    #     orig_snr = result['original_metrics']['snr_estimate']
    #     proc_snr = result['processed_metrics']['snr_estimate']
    #     snr_improvement = (proc_snr - orig_snr) / orig_snr * 100
        
    #     orig_contrast = result['original_metrics']['contrast']
    #     proc_contrast = result['processed_metrics']['contrast']
    #     contrast_improvement = (proc_contrast - orig_contrast) / orig_contrast * 100
        
    #     snr_improvements.append(snr_improvement)
    #     contrast_improvements.append(contrast_improvement)
        

    




if __name__ == "__main__":
    main()