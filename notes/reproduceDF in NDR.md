Find the video for Fig 4 in NDR paper. => seg026 in test. We'll cut and only keep first 100 frames.
Run preprocess_our

If you run kniectfusion only on seq026 (commit 656dffe: Automatic Commit: Thu 18 May 2023 06:42:49 PM CDT.) then the initial result will be ok. But with wrapping, the whole thing collapse. So there is a bug in this code.
Then we will segment the video using tool like MiVOS, instead of depth cutting.
