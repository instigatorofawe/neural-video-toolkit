#!/usr/bin/env python
from argparse import ArgumentParser

import ffmpeg
import numpy
import torch

from src import get_video_info
from src.ESRGAN import extract_model_parameters, architecture

if __name__ == '__main__':
    # Parse command line arguments
    parser = ArgumentParser(description="""Upscale a video file using an image super resolution method.
    Uses ffmpeg for encoding and decoding.""")

    parser.add_argument("file", metavar="input_filename", type=str, help="Filename of input video file.")

    parser.add_argument("-o", "--output", metavar="output_filename", type=str,
                        help="Filename of output video file. Defaults to (input_filename)_upscaled.")
    parser.add_argument("-m", "--model", metavar="model_filename", type=str,
                        help="Filename of model weights.", default="models/2x_FilmFrames_0.5_677000_G.pth")
    parser.add_argument("-a", "--arch", metavar="architecture", type=str,
                        help="Architecture for upscaling method. Supported options: esrgan",
                        default="esrgan")

    args = parser.parse_args()

    # Different architectures need to be handled differently
    model = None
    upscale = 1

    if args.arch == "esrgan":
        state_dict = torch.load(args.model)
        in_nc, out_nc, nf, nb, upscale = extract_model_parameters(state_dict)
        model = architecture.RRDBNet(in_nc, out_nc, nf, nb, gc=32, upscale=upscale, norm_type=None,
                                     act_type='leakyrelu', mode='CNA', res_scale=1, upsample_mode='upconv')
        model.load_state_dict(state_dict, strict=True)

        # Turn on evaluation mode and turn off gradient computation in pytorch
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to('cuda')
    else:
        print("Architecture not supported!")
        exit(1)

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
        .output(output_filename, pix_fmt="yuv420p", **{"c:v": "libx264", "crf": "10"})
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    while True:
        in_bytes = input_process.stdout.read(width*height*3)
        if not in_bytes:
            break
        img = numpy.frombuffer(in_bytes, 'uint8').reshape([height, width, 3])
        img = img * 1. / numpy.iinfo(img.dtype).max

        # Will have to refactor this code to support Tensorflow models
        img = torch.from_numpy(numpy.transpose(img, (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to('cuda')

        with torch.no_grad():
            output = model(img_LR).data.squeeze(0).float().cpu().clamp_(0, 1).numpy()

        output = numpy.transpose(output, (1, 2, 0))
        output = (output * 255.0).round().astype('uint8')

        output_process.stdin.write(output.tobytes())

    output_process.stdin.close()
    input_process.wait()
    output_process.wait()

