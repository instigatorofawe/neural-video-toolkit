#!/usr/bin/env python
from argparse import ArgumentParser

import ffmpeg
import numpy
import tensorflow
import torch

from src import get_video_info, ESRGAN, CAR
from src.CAR.edsr import EDSR
from src.ESRGAN import architecture
from src.subpixel_cnn import load_subpixel_cnn

if __name__ == '__main__':
    # Parse command line arguments
    parser = ArgumentParser(description="""Upscale a video file using an image super resolution method.
    Uses ffmpeg for encoding and decoding.""")

    parser.add_argument("file", metavar="input_filename", type=str, help="Filename of input video file.")

    parser.add_argument("-o", "--output", metavar="output_filename", type=str,
                        help="Filename of output video file. Defaults to (input_filename)_upscaled.")
    parser.add_argument("-m", "--model", metavar="model_filename", type=str,
                        help="Filename of model weights.")
    parser.add_argument("-a", "--arch", metavar="architecture", type=str,
                        help="""Architecture for upscaling method. Supported options: esrgan, subpixel_cnn, car. 
                        Default: esrgan""", default="esrgan")
    args = parser.parse_args(["-m", "./models/2x_filmframes.pth", "./data/ep14_ed_full.mkv"])
    # args = parser.parse_args()

    # Different architectures need to be handled differently
    model = None
    upscale = 1  # Coefficient by which we're scaling the output video
    channels = 3  # Number of color channels that model operates on

    if args.arch == "esrgan":
        state_dict = torch.load(args.model)
        in_nc, out_nc, nf, nb, upscale = ESRGAN.extract_model_parameters(state_dict)
        model = architecture.RRDBNet(in_nc, out_nc, nf, nb, gc=32, upscale=upscale, norm_type=None,
                                     act_type='leakyrelu', mode='CNA', res_scale=1, upsample_mode='upconv')
        model.load_state_dict(state_dict, strict=True)
        # Turn on evaluation mode and turn off gradient computation in pytorch
        model.eval()
        model = model.to('cuda')
    elif args.arch == "subpixel_cnn":
        channels, upscale, model = load_subpixel_cnn(args.model)
    elif args.arch == "car":
        state_dict = torch.load(args.model)
        upscale = CAR.extract_model_parameters(state_dict)
        model = EDSR(32, 256, scale=upscale)
        model = torch.nn.DataParallel(model, [0])
        model.load_state_dict(state_dict)
        model.eval()
    else:
        print("Architecture not supported!")
        exit(1)

    # Disable computation of gradients on PyTorch, as we are evaluating, not training
    torch.set_grad_enabled(False)

    input_filename = str(args.file)
    # Get filename prefix without extension
    if args.output is not None:
        output_filename = args.output
    else:
        prefix = input_filename.split(".")[0]
        output_filename = f"%s_upscaled.mkv" % prefix

    # Extract video information using ffmpeg.probe()
    info = ffmpeg.probe(input_filename)
    width, height, framerate = get_video_info(info)

    # Set up decoding and encoding ffmpeg processes
    input_process = (
        ffmpeg
            .input(input_filename)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run_async(pipe_stdout=True)
    )

    # TODO: add additional encoding options instead of hard coding compression settings
    output_process = (
        ffmpeg
            .input("pipe:", format="rawvideo", pix_fmt="rgb24", s=f"%dx%d" % (width * upscale, height * upscale),
                   framerate=framerate)
            .output(output_filename, pix_fmt="yuv444p10le",
                    vf="scale=in_color_matrix=auto:in_range=auto:out_color_matrix=bt709:out_range=tv",
                    **{"c:v": "libx265", "crf": "10", "colorspace:v": "bt709", "color_primaries:v": "bt709",
                       "color_trc:v": "bt709", "color_range:v": "tv"})
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )

    last_frame = None
    last_output = None
    n_duplicate_frames = 0
    while True:
        in_bytes = input_process.stdout.read(width * height * 3)
        if not in_bytes:
            break

        # Detect duplicates, and use the cached output if there is a duplicate frame
        if last_frame is not None and last_frame == in_bytes:
            output_process.stdin.write(last_output.tobytes())
            n_duplicate_frames += 1
            continue

        # Separate handling for separate model architectures
        if args.arch == "esrgan" or args.arch == "car":
            img = numpy.frombuffer(in_bytes, 'uint8').reshape([height, width, 3])
            img = img * 1. / numpy.iinfo(img.dtype).max
            img = torch.from_numpy(numpy.transpose(img, (2, 0, 1))).float()
            img_LR = img.unsqueeze(0)
            img_LR = img_LR.to('cuda')
            with torch.no_grad():
                output = model(img_LR).data.squeeze(0).float().cpu().clamp_(0, 1).numpy()
            output = numpy.transpose(output, (1, 2, 0))

        elif args.arch == "subpixel_cnn":
            img = numpy.frombuffer(in_bytes, 'uint8').reshape([1, height, width, 3])
            img = img * 1. / numpy.iinfo(img.dtype).max
            # One filter is run on each channel
            img = img.transpose((3, 1, 2, 0)).astype('float32')
            output = tensorflow.clip_by_value(model(img), 0, 1).numpy()
            output = output.transpose((3, 1, 2, 0))

        # noinspection PyUnboundLocalVariable
        output = (output * 255.0).round().astype('uint8')
        output_process.stdin.write(output.tobytes())

        last_frame = in_bytes
        last_output = output

    output_process.stdin.close()
    input_process.wait()
    output_process.wait()

    print(f"%d duplicate frames detected" % n_duplicate_frames)
