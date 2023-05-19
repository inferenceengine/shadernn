# Copyright (C) 2020 - 2022 OPPO. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import cv2, argparse, os
import tensorflow as tf

def compute_psnr(image1, image2):
    return cv2.PSNR(image1, image2)

def compute_ssim(image1, image2):
    im1 = tf.image.convert_image_dtype(image1, tf.uint8)
    im2 = tf.image.convert_image_dtype(image2, tf.uint8)
    ssim1 = tf.image.ssim(im1, im2, max_val=255, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
    return ssim1

def compareImages(image1, image2, outDir):
    try:
        img1 = cv2.imread(image1)
        img2 = cv2.imread(image2)
    except:
        print("Error: cannot read image files")
        return
    
    psnr = compute_psnr(img1, img2)
    ssim = compute_ssim(img1, img2)

    print("\n\n#######PSNR: {}".format(psnr))
    print("#######SSIM: {}".format(ssim))
    diffFile = os.path.join(outDir, "diff.png")
    print("Writing Diff Output to {}\n".format(diffFile))
    diff = 255 - cv2.absdiff(img1, img2)
    cv2.imwrite(diffFile, diff)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image1", help="Path of 1st Image", required=True)
    parser.add_argument("--image2", help="Path of 2nd Image", required=True)
    parser.add_argument("--outdir", help="Output Directory", required=True)
    args = parser.parse_args()
    
    image1  = str(args.image1)
    image2  = str(args.image2)
    outDir  = str(args.outdir)
    compareImages(image1, image2, outDir)
