---
layout: post
title: Translating Images into Maps
date: 2023-08-06 00:00:00
img: vision/fusion/translating_images_into_maps/0.png
categories: [vision-fusion] 
tags: [nvautonet, multi-camera fusion] # add tag
---

<br>

[fusion 관련 글 목차](https://gaussian37.github.io/vision-fusion-table/)

<br>

- 논문 : https://arxiv.org/abs/2110.00966
- 깃헙 : https://github.com/avishkarsaha/translating-images-into-maps

<div class="demo_3d" style="">
    <table style="width: 100%"><tbody><tr style="text-align:center;"><td width="50%">Scene</td><td>Image</td></tr></tbody></table>
    <div id="3d_container"><canvas width="500" height="166"></canvas>
    </div>
    <div class="caption">
    <em>Left</em>: scene with camera and viewing volume.  Virtual image plane is shown in yellow.   <em>Right</em>: camera's image.</div>
    <div id="demo_controls" class="ui-tabs ui-widget ui-widget-content ui-corner-all">
        <ul class="ui-tabs-nav ui-helper-reset ui-helper-clearfix ui-widget-header ui-corner-all">
            <li class="ui-state-default ui-corner-top"><a href="#extrinsic-world-controls">Extrinsic (World)</a></li>
            <li class="ui-state-default ui-corner-top"><a href="#extrinsic-camera-controls">Extr. (Camera)</a></li>
            <li class="ui-state-default ui-corner-top"><a href="#extrinsic-lookat-controls">Extr. ("Look-at")</a></li>
            <li class="ui-state-default ui-corner-top ui-tabs-selected ui-state-active"><a href="#intrinsic-controls">Intrinsic</a></li>
        </ul>
        <div id="extrinsic-world-controls" class="ui-tabs-panel ui-widget-content ui-corner-bottom ui-tabs-hide">
            <div class="slider-control">
                <div class="slider ui-slider ui-slider-horizontal ui-widget ui-widget-content ui-corner-all" id="world_x_slider">
                <a class="ui-slider-handle ui-state-default ui-corner-all" href="#" style="left: 52.1667%;"></a></div>
                <div class="slider-label">
                <span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" id="MathJax-Element-1-Frame" tabindex="0" style="position: relative;" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mi mathvariant=&quot;bold-italic&quot;>t</mi><mi>x</mi></msub></math>" role="presentation"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-1" style="width: 1.129em; display: inline-block;"><span style="display: inline-block; position: relative; width: 0.904em; height: 0px; font-size: 119%;"><span style="position: absolute; clip: rect(1.429em, 1000.9em, 2.705em, -999.996em); top: -2.322em; left: 0em;"><span class="mrow" id="MathJax-Span-2"><span class="msubsup" id="MathJax-Span-3"><span style="display: inline-block; position: relative; width: 0.904em; height: 0px;"><span style="position: absolute; clip: rect(3.08em, 1000.38em, 4.205em, -999.996em); top: -3.973em; left: 0em;"><span class="mi" id="MathJax-Span-4" style="font-family: MathJax_Math-bold-italic;">t</span><span style="display: inline-block; width: 0px; height: 3.98em;"></span></span><span style="position: absolute; top: -3.823em; left: 0.454em;"><span class="mi" id="MathJax-Span-5" style="font-size: 70.7%; font-family: MathJax_Math-italic;">x</span><span style="display: inline-block; width: 0px; height: 3.98em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.33em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.263em; border-left: 0px solid; width: 0px; height: 1.165em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi mathvariant="bold-italic">t</mi><mi>x</mi></msub></math></span></span><script type="math/tex" id="MathJax-Element-1">\boldsymbol{t}_x</script>
                </div>
                <div class="clearer"></div>
            </div>
            <div class="slider-control">
                <div class="slider ui-slider ui-slider-horizontal ui-widget ui-widget-content ui-corner-all" id="world_y_slider">
                <a class="ui-slider-handle ui-state-default ui-corner-all" href="#" style="left: 51.3333%;"></a></div>
                <div class="slider-label">
                <span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" id="MathJax-Element-2-Frame" tabindex="0" style="position: relative;" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mi mathvariant=&quot;bold-italic&quot;>t</mi><mi>y</mi></msub></math>" role="presentation"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-6" style="width: 0.979em; display: inline-block;"><span style="display: inline-block; position: relative; width: 0.829em; height: 0px; font-size: 119%;"><span style="position: absolute; clip: rect(1.429em, 1000.83em, 2.855em, -999.996em); top: -2.322em; left: 0em;"><span class="mrow" id="MathJax-Span-7"><span class="msubsup" id="MathJax-Span-8"><span style="display: inline-block; position: relative; width: 0.829em; height: 0px;"><span style="position: absolute; clip: rect(3.08em, 1000.38em, 4.205em, -999.996em); top: -3.973em; left: 0em;"><span class="mi" id="MathJax-Span-9" style="font-family: MathJax_Math-bold-italic;">t</span><span style="display: inline-block; width: 0px; height: 3.98em;"></span></span><span style="position: absolute; top: -3.823em; left: 0.454em;"><span class="mi" id="MathJax-Span-10" style="font-size: 70.7%; font-family: MathJax_Math-italic;">y<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.004em;"></span></span><span style="display: inline-block; width: 0px; height: 3.98em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.33em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.442em; border-left: 0px solid; width: 0px; height: 1.344em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi mathvariant="bold-italic">t</mi><mi>y</mi></msub></math></span></span><script type="math/tex" id="MathJax-Element-2">\boldsymbol{t}_y</script>
                </div>
                <div class="clearer"></div>
            </div>
            <div class="slider-control">
                <div class="slider ui-slider ui-slider-horizontal ui-widget ui-widget-content ui-corner-all" id="world_z_slider">
                <a class="ui-slider-handle ui-state-default ui-corner-all" href="#" style="left: 50.7692%;"></a></div>
                <div class="slider-label">
                <span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" id="MathJax-Element-3-Frame" tabindex="0" style="position: relative;" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mi mathvariant=&quot;bold-italic&quot;>t</mi><mi>z</mi></msub></math>" role="presentation"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-11" style="width: 0.979em; display: inline-block;"><span style="display: inline-block; position: relative; width: 0.829em; height: 0px; font-size: 119%;"><span style="position: absolute; clip: rect(1.429em, 1000.83em, 2.705em, -999.996em); top: -2.322em; left: 0em;"><span class="mrow" id="MathJax-Span-12"><span class="msubsup" id="MathJax-Span-13"><span style="display: inline-block; position: relative; width: 0.829em; height: 0px;"><span style="position: absolute; clip: rect(3.08em, 1000.38em, 4.205em, -999.996em); top: -3.973em; left: 0em;"><span class="mi" id="MathJax-Span-14" style="font-family: MathJax_Math-bold-italic;">t</span><span style="display: inline-block; width: 0px; height: 3.98em;"></span></span><span style="position: absolute; top: -3.823em; left: 0.454em;"><span class="mi" id="MathJax-Span-15" style="font-size: 70.7%; font-family: MathJax_Math-italic;">z<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.004em;"></span></span><span style="display: inline-block; width: 0px; height: 3.98em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.33em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.263em; border-left: 0px solid; width: 0px; height: 1.165em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi mathvariant="bold-italic">t</mi><mi>z</mi></msub></math></span></span><script type="math/tex" id="MathJax-Element-3">\boldsymbol{t}_z</script>
                </div>
                <div class="clearer"></div>
            </div>
            <div class="slider-control">
                <div class="slider ui-slider ui-slider-horizontal ui-widget ui-widget-content ui-corner-all" id="world_rx_slider">
                <a class="ui-slider-handle ui-state-default ui-corner-all" href="#" style="left: 50.6112%;"></a></div>
                <div class="slider-label">
                x-Rotation
                </div>
                <div class="clearer"></div>
            </div>
            <div class="slider-control">
                <div class="slider ui-slider ui-slider-horizontal ui-widget ui-widget-content ui-corner-all" id="world_ry_slider">
                <a class="ui-slider-handle ui-state-default ui-corner-all" href="#" style="left: 51.5661%;"></a></div>
                <div class="slider-label">
                y-Rotation
                </div>
                <div class="clearer"></div>
            </div>
            <div class="slider-control">
                <div class="slider ui-slider ui-slider-horizontal ui-widget ui-widget-content ui-corner-all" id="world_rz_slider">
                <a class="ui-slider-handle ui-state-default ui-corner-all" href="#" style="left: 49.6562%;"></a></div>
                <div class="slider-label">
                z-Rotation
                </div>
                <div class="clearer"></div>
            </div>
            <p>Adjust extrinsic parameters above.</p>

            <p>This is a "world-centric" parameterization.  These parameters describe how the <em>world</em> changes relative to the <em>camera</em>.  These parameters correspond directly to entries in the extrinsic camera matrix.</p>
            
            <p>As you adjust these parameters, note how the camera moves in the world (left pane) and contrast with the "camera-centric" parameterization:</p>
            <ul>
            <li>Rotating affects the camera's position (the blue box).</li>
            <li>The direction of camera motion depends on the current rotation.</li>
            <li>Positive rotations move the camera clockwise (or equivalently, rotate the world counter-clockwise).</li>
            </ul>

            <p>Also note how the image is affected (right pane):</p>

            <ul>
            <li>Rotating never moves the world origin (red ball).</li>
            <li>Changing <span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" id="MathJax-Element-4-Frame" tabindex="0" style="position: relative;" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mi>t</mi><mi>x</mi></msub></math>" role="presentation"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-16" style="width: 1.066em; display: inline-block;"><span style="display: inline-block; position: relative; width: 0.854em; height: 0px; font-size: 126%;"><span style="position: absolute; clip: rect(1.35em, 1000.85em, 2.555em, -999.996em); top: -2.193em; left: 0em;"><span class="mrow" id="MathJax-Span-17"><span class="msubsup" id="MathJax-Span-18"><span style="display: inline-block; position: relative; width: 0.854em; height: 0px;"><span style="position: absolute; clip: rect(3.121em, 1000.36em, 4.184em, -999.996em); top: -3.965em; left: 0em;"><span class="mi" id="MathJax-Span-19" style="font-family: MathJax_Math-italic;">t</span><span style="display: inline-block; width: 0px; height: 3.972em;"></span></span><span style="position: absolute; top: -3.823em; left: 0.358em;"><span class="mi" id="MathJax-Span-20" style="font-size: 70.7%; font-family: MathJax_Math-italic;">x</span><span style="display: inline-block; width: 0px; height: 3.972em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.2em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.263em; border-left: 0px solid; width: 0px; height: 1.165em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>t</mi><mi>x</mi></msub></math></span></span><script type="math/tex" id="MathJax-Element-4">t_x</script> always moves the spheres horizontally, regardless of rotation. </li>
            <li>Increasing <span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" id="MathJax-Element-5-Frame" tabindex="0" style="position: relative;" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mi>t</mi><mi>z</mi></msub></math>" role="presentation"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-21" style="width: 0.996em; display: inline-block;"><span style="display: inline-block; position: relative; width: 0.783em; height: 0px; font-size: 126%;"><span style="position: absolute; clip: rect(1.35em, 1000.78em, 2.555em, -999.996em); top: -2.193em; left: 0em;"><span class="mrow" id="MathJax-Span-22"><span class="msubsup" id="MathJax-Span-23"><span style="display: inline-block; position: relative; width: 0.783em; height: 0px;"><span style="position: absolute; clip: rect(3.121em, 1000.36em, 4.184em, -999.996em); top: -3.965em; left: 0em;"><span class="mi" id="MathJax-Span-24" style="font-family: MathJax_Math-italic;">t</span><span style="display: inline-block; width: 0px; height: 3.972em;"></span></span><span style="position: absolute; top: -3.823em; left: 0.358em;"><span class="mi" id="MathJax-Span-25" style="font-size: 70.7%; font-family: MathJax_Math-italic;">z<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.004em;"></span></span><span style="display: inline-block; width: 0px; height: 3.972em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.2em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.263em; border-left: 0px solid; width: 0px; height: 1.165em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>t</mi><mi>z</mi></msub></math></span></span><script type="math/tex" id="MathJax-Element-5">t_z</script> always moves the camera closer to the world origin. </li>
            </ul>
        </div>
        <div id="extrinsic-camera-controls" class="ui-tabs-panel ui-widget-content ui-corner-bottom ui-tabs-hide">
            <div class="slider-control">
                <div class="slider ui-slider ui-slider-horizontal ui-widget ui-widget-content ui-corner-all" id="camera_x_slider">
                <a class="ui-slider-handle ui-state-default ui-corner-all" href="#" style="left: 47.8333%;"></a></div>
                <div class="slider-label">
                <span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" id="MathJax-Element-6-Frame" tabindex="0" style="position: relative;" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mi>C</mi><mi>x</mi></msub></math>" role="presentation"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-26" style="width: 1.461em; display: inline-block;"><span style="display: inline-block; position: relative; width: 1.201em; height: 0px; font-size: 120%;"><span style="position: absolute; clip: rect(1.253em, 1001.2em, 2.451em, -999.997em); top: -2.133em; left: 0em;"><span class="mrow" id="MathJax-Span-27"><span class="msubsup" id="MathJax-Span-28"><span style="display: inline-block; position: relative; width: 1.201em; height: 0px;"><span style="position: absolute; clip: rect(3.128em, 1000.78em, 4.169em, -999.997em); top: -4.008em; left: 0em;"><span class="mi" id="MathJax-Span-29" style="font-family: MathJax_Math-italic;">C<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.055em;"></span></span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span><span style="position: absolute; top: -3.852em; left: 0.732em;"><span class="mi" id="MathJax-Span-30" style="font-size: 70.7%; font-family: MathJax_Math-italic;">x</span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.138em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.247em; border-left: 0px solid; width: 0px; height: 1.191em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>C</mi><mi>x</mi></msub></math></span></span><script type="math/tex" id="MathJax-Element-6">C_x</script>
                </div>
                <div class="clearer"></div>
            </div>
            <div class="slider-control">
                <div class="slider ui-slider ui-slider-horizontal ui-widget ui-widget-content ui-corner-all" id="camera_y_slider">
                <a class="ui-slider-handle ui-state-default ui-corner-all" href="#" style="left: 48.6667%;"></a></div>
                <div class="slider-label">
                <span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" id="MathJax-Element-7-Frame" tabindex="0" style="position: relative;" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mi>C</mi><mi>y</mi></msub></math>" role="presentation"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-31" style="width: 1.409em; display: inline-block;"><span style="display: inline-block; position: relative; width: 1.148em; height: 0px; font-size: 120%;"><span style="position: absolute; clip: rect(1.253em, 1001.15em, 2.607em, -999.997em); top: -2.133em; left: 0em;"><span class="mrow" id="MathJax-Span-32"><span class="msubsup" id="MathJax-Span-33"><span style="display: inline-block; position: relative; width: 1.148em; height: 0px;"><span style="position: absolute; clip: rect(3.128em, 1000.78em, 4.169em, -999.997em); top: -4.008em; left: 0em;"><span class="mi" id="MathJax-Span-34" style="font-family: MathJax_Math-italic;">C<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.055em;"></span></span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span><span style="position: absolute; top: -3.852em; left: 0.732em;"><span class="mi" id="MathJax-Span-35" style="font-size: 70.7%; font-family: MathJax_Math-italic;">y<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.003em;"></span></span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.138em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.434em; border-left: 0px solid; width: 0px; height: 1.316em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>C</mi><mi>y</mi></msub></math></span></span><script type="math/tex" id="MathJax-Element-7">C_y</script>
                </div>
                <div class="clearer"></div>
            </div>
            <div class="slider-control">
                <div class="slider ui-slider ui-slider-horizontal ui-widget ui-widget-content ui-corner-all" id="camera_z_slider">
                <a class="ui-slider-handle ui-state-default ui-corner-all" href="#" style="left: 49.2308%;"></a></div>
                <div class="slider-label">
                <span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" id="MathJax-Element-8-Frame" tabindex="0" style="position: relative;" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mi>C</mi><mi>z</mi></msub></math>" role="presentation"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-36" style="width: 1.409em; display: inline-block;"><span style="display: inline-block; position: relative; width: 1.148em; height: 0px; font-size: 120%;"><span style="position: absolute; clip: rect(1.253em, 1001.15em, 2.451em, -999.997em); top: -2.133em; left: 0em;"><span class="mrow" id="MathJax-Span-37"><span class="msubsup" id="MathJax-Span-38"><span style="display: inline-block; position: relative; width: 1.148em; height: 0px;"><span style="position: absolute; clip: rect(3.128em, 1000.78em, 4.169em, -999.997em); top: -4.008em; left: 0em;"><span class="mi" id="MathJax-Span-39" style="font-family: MathJax_Math-italic;">C<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.055em;"></span></span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span><span style="position: absolute; top: -3.852em; left: 0.732em;"><span class="mi" id="MathJax-Span-40" style="font-size: 70.7%; font-family: MathJax_Math-italic;">z<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.003em;"></span></span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.138em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.247em; border-left: 0px solid; width: 0px; height: 1.191em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>C</mi><mi>z</mi></msub></math></span></span><script type="math/tex" id="MathJax-Element-8">C_z</script>
                </div>
                <div class="clearer"></div>
            </div>
            <div class="slider-control">
                <div class="slider ui-slider ui-slider-horizontal ui-widget ui-widget-content ui-corner-all" id="camera_rx_slider">
                <a class="ui-slider-handle ui-state-default ui-corner-all" href="#" style="left: 49.6562%;"></a></div>
                <div class="slider-label">
                x-Rotation
                </div>
                <div class="clearer"></div>
            </div>
            <div class="slider-control">
                <div class="slider ui-slider ui-slider-horizontal ui-widget ui-widget-content ui-corner-all" id="camera_ry_slider">
                <a class="ui-slider-handle ui-state-default ui-corner-all" href="#" style="left: 48.7013%;"></a></div>
                <div class="slider-label">
                y-Rotation
                </div>
                <div class="clearer"></div>
            </div>
            <div class="slider-control">
                <div class="slider ui-slider ui-slider-horizontal ui-widget ui-widget-content ui-corner-all" id="camera_rz_slider">
                <a class="ui-slider-handle ui-state-default ui-corner-all" href="#" style="left: 49.6562%;"></a></div>
                <div class="slider-label">
                z-Rotation
                </div>
                <div class="clearer"></div>
            </div>
            <p>Adjust extrinsic parameters above.</p>

            <p>This is a "camera-centric" parameterization, which describes how the <em>camera</em> changes relative to the <em>world</em>.  These parameters correspond to elements of the <em>inverse</em> extrinsic camera matrix.</p>
            
            <p>As you adjust these parameters, note how the camera moves in the world (left pane) and contrast with the "world-centric" parameterization:</p>
            <ul>
            <li>Rotation occurs about the camera's position (the blue box).</li>
            <li>The direction of camera motion is independent of the current rotation.</li>
            <li>A positive rotation rotates the camera counter-clockwise (or equivalently, rotates the world clockwise).</li>
            <li>Increasing <span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" id="MathJax-Element-9-Frame" tabindex="0" style="position: relative;" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mi>C</mi><mi>y</mi></msub></math>" role="presentation"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-41" style="width: 1.409em; display: inline-block;"><span style="display: inline-block; position: relative; width: 1.148em; height: 0px; font-size: 120%;"><span style="position: absolute; clip: rect(1.253em, 1001.15em, 2.607em, -999.997em); top: -2.133em; left: 0em;"><span class="mrow" id="MathJax-Span-42"><span class="msubsup" id="MathJax-Span-43"><span style="display: inline-block; position: relative; width: 1.148em; height: 0px;"><span style="position: absolute; clip: rect(3.128em, 1000.78em, 4.169em, -999.997em); top: -4.008em; left: 0em;"><span class="mi" id="MathJax-Span-44" style="font-family: MathJax_Math-italic;">C<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.055em;"></span></span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span><span style="position: absolute; top: -3.852em; left: 0.732em;"><span class="mi" id="MathJax-Span-45" style="font-size: 70.7%; font-family: MathJax_Math-italic;">y<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.003em;"></span></span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.138em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.434em; border-left: 0px solid; width: 0px; height: 1.316em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>C</mi><mi>y</mi></msub></math></span></span><script type="math/tex" id="MathJax-Element-9">C_y</script> always moves the camera toward the sky, regardless of rotation. </li>
            </ul>

            <p>Also note how the image is affected (right pane):</p>

            <ul>
            <li>Rotating around y moves both spheres horizontally.</li>
            <li>With different rotations, changing <span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" id="MathJax-Element-10-Frame" tabindex="0" style="position: relative;" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mi>C</mi><mi>x</mi></msub></math>" role="presentation"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-46" style="width: 1.461em; display: inline-block;"><span style="display: inline-block; position: relative; width: 1.201em; height: 0px; font-size: 120%;"><span style="position: absolute; clip: rect(1.253em, 1001.2em, 2.451em, -999.997em); top: -2.133em; left: 0em;"><span class="mrow" id="MathJax-Span-47"><span class="msubsup" id="MathJax-Span-48"><span style="display: inline-block; position: relative; width: 1.201em; height: 0px;"><span style="position: absolute; clip: rect(3.128em, 1000.78em, 4.169em, -999.997em); top: -4.008em; left: 0em;"><span class="mi" id="MathJax-Span-49" style="font-family: MathJax_Math-italic;">C<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.055em;"></span></span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span><span style="position: absolute; top: -3.852em; left: 0.732em;"><span class="mi" id="MathJax-Span-50" style="font-size: 70.7%; font-family: MathJax_Math-italic;">x</span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.138em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.247em; border-left: 0px solid; width: 0px; height: 1.191em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>C</mi><mi>x</mi></msub></math></span></span><script type="math/tex" id="MathJax-Element-10">C_x</script> moves the spheres in different directions. </li>
            </ul>
        </div>

        <div id="extrinsic-lookat-controls" class="ui-tabs-panel ui-widget-content ui-corner-bottom ui-tabs-hide">
            <div class="slider-control">
                <div class="slider ui-slider ui-slider-horizontal ui-widget ui-widget-content ui-corner-all" id="lookat_x_slider">
                <a class="ui-slider-handle ui-state-default ui-corner-all" href="#" style="left: 47.8333%;"></a></div>
                <div class="slider-label">
                <span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" id="MathJax-Element-11-Frame" tabindex="0" style="position: relative;" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mi>C</mi><mi>x</mi></msub></math>" role="presentation"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-51" style="width: 1.461em; display: inline-block;"><span style="display: inline-block; position: relative; width: 1.201em; height: 0px; font-size: 120%;"><span style="position: absolute; clip: rect(1.253em, 1001.2em, 2.451em, -999.997em); top: -2.133em; left: 0em;"><span class="mrow" id="MathJax-Span-52"><span class="msubsup" id="MathJax-Span-53"><span style="display: inline-block; position: relative; width: 1.201em; height: 0px;"><span style="position: absolute; clip: rect(3.128em, 1000.78em, 4.169em, -999.997em); top: -4.008em; left: 0em;"><span class="mi" id="MathJax-Span-54" style="font-family: MathJax_Math-italic;">C<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.055em;"></span></span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span><span style="position: absolute; top: -3.852em; left: 0.732em;"><span class="mi" id="MathJax-Span-55" style="font-size: 70.7%; font-family: MathJax_Math-italic;">x</span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.138em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.247em; border-left: 0px solid; width: 0px; height: 1.191em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>C</mi><mi>x</mi></msub></math></span></span><script type="math/tex" id="MathJax-Element-11">C_x</script>
                </div>
                <div class="clearer"></div>
            </div>
            <div class="slider-control">
                <div class="slider ui-slider ui-slider-horizontal ui-widget ui-widget-content ui-corner-all" id="lookat_y_slider">
                <a class="ui-slider-handle ui-state-default ui-corner-all" href="#" style="left: 48.6667%;"></a></div>
                <div class="slider-label">
                <span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" id="MathJax-Element-12-Frame" tabindex="0" style="position: relative;" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mi>C</mi><mi>y</mi></msub></math>" role="presentation"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-56" style="width: 1.409em; display: inline-block;"><span style="display: inline-block; position: relative; width: 1.148em; height: 0px; font-size: 120%;"><span style="position: absolute; clip: rect(1.253em, 1001.15em, 2.607em, -999.997em); top: -2.133em; left: 0em;"><span class="mrow" id="MathJax-Span-57"><span class="msubsup" id="MathJax-Span-58"><span style="display: inline-block; position: relative; width: 1.148em; height: 0px;"><span style="position: absolute; clip: rect(3.128em, 1000.78em, 4.169em, -999.997em); top: -4.008em; left: 0em;"><span class="mi" id="MathJax-Span-59" style="font-family: MathJax_Math-italic;">C<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.055em;"></span></span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span><span style="position: absolute; top: -3.852em; left: 0.732em;"><span class="mi" id="MathJax-Span-60" style="font-size: 70.7%; font-family: MathJax_Math-italic;">y<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.003em;"></span></span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.138em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.434em; border-left: 0px solid; width: 0px; height: 1.316em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>C</mi><mi>y</mi></msub></math></span></span><script type="math/tex" id="MathJax-Element-12">C_y</script>
                </div>
                <div class="clearer"></div>
            </div>
            <div class="slider-control">
                <div class="slider ui-slider ui-slider-horizontal ui-widget ui-widget-content ui-corner-all" id="lookat_z_slider">
                <a class="ui-slider-handle ui-state-default ui-corner-all" href="#" style="left: 49.2308%;"></a></div>
                <div class="slider-label">
                <span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" id="MathJax-Element-13-Frame" tabindex="0" style="position: relative;" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mi>C</mi><mi>z</mi></msub></math>" role="presentation"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-61" style="width: 1.409em; display: inline-block;"><span style="display: inline-block; position: relative; width: 1.148em; height: 0px; font-size: 120%;"><span style="position: absolute; clip: rect(1.253em, 1001.15em, 2.451em, -999.997em); top: -2.133em; left: 0em;"><span class="mrow" id="MathJax-Span-62"><span class="msubsup" id="MathJax-Span-63"><span style="display: inline-block; position: relative; width: 1.148em; height: 0px;"><span style="position: absolute; clip: rect(3.128em, 1000.78em, 4.169em, -999.997em); top: -4.008em; left: 0em;"><span class="mi" id="MathJax-Span-64" style="font-family: MathJax_Math-italic;">C<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.055em;"></span></span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span><span style="position: absolute; top: -3.852em; left: 0.732em;"><span class="mi" id="MathJax-Span-65" style="font-size: 70.7%; font-family: MathJax_Math-italic;">z<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.003em;"></span></span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.138em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.247em; border-left: 0px solid; width: 0px; height: 1.191em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>C</mi><mi>z</mi></msub></math></span></span><script type="math/tex" id="MathJax-Element-13">C_z</script>
                </div>
                <div class="clearer"></div>
            </div>
            <div class="slider-control">
                <div class="slider ui-slider ui-slider-horizontal ui-widget ui-widget-content ui-corner-all" id="lookat_px_slider">
                <a class="ui-slider-handle ui-state-default ui-corner-all" href="#" style="left: 48.745%;"></a></div>
                <div class="slider-label">
                <span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" id="MathJax-Element-14-Frame" tabindex="0" style="position: relative;" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mi>p</mi><mi>x</mi></msub></math>" role="presentation"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-66" style="width: 1.201em; display: inline-block;"><span style="display: inline-block; position: relative; width: 0.992em; height: 0px; font-size: 120%;"><span style="position: absolute; clip: rect(1.565em, 1000.99em, 2.503em, -999.997em); top: -2.133em; left: 0em;"><span class="mrow" id="MathJax-Span-67"><span class="msubsup" id="MathJax-Span-68"><span style="display: inline-block; position: relative; width: 0.992em; height: 0px;"><span style="position: absolute; clip: rect(3.44em, 1000.52em, 4.378em, -999.997em); top: -4.008em; left: 0em;"><span class="mi" id="MathJax-Span-69" style="font-family: MathJax_Math-italic;">p</span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span><span style="position: absolute; top: -3.852em; left: 0.523em;"><span class="mi" id="MathJax-Span-70" style="font-size: 70.7%; font-family: MathJax_Math-italic;">x</span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.138em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.309em; border-left: 0px solid; width: 0px; height: 0.878em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>p</mi><mi>x</mi></msub></math></span></span><script type="math/tex" id="MathJax-Element-14">p_x</script>
                </div>
                <div class="clearer"></div>
            </div>
            <div class="slider-control">
                <div class="slider ui-slider ui-slider-horizontal ui-widget ui-widget-content ui-corner-all" id="lookat_py_slider">
                <a class="ui-slider-handle ui-state-default ui-corner-all" href="#" style="left: 48.425%;"></a></div>
                <div class="slider-label">
                <span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" id="MathJax-Element-15-Frame" tabindex="0" style="position: relative;" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mi>p</mi><mi>y</mi></msub></math>" role="presentation"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-71" style="width: 1.148em; display: inline-block;"><span style="display: inline-block; position: relative; width: 0.94em; height: 0px; font-size: 120%;"><span style="position: absolute; clip: rect(1.565em, 1000.94em, 2.607em, -999.997em); top: -2.133em; left: 0em;"><span class="mrow" id="MathJax-Span-72"><span class="msubsup" id="MathJax-Span-73"><span style="display: inline-block; position: relative; width: 0.94em; height: 0px;"><span style="position: absolute; clip: rect(3.44em, 1000.52em, 4.378em, -999.997em); top: -4.008em; left: 0em;"><span class="mi" id="MathJax-Span-74" style="font-family: MathJax_Math-italic;">p</span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span><span style="position: absolute; top: -3.852em; left: 0.523em;"><span class="mi" id="MathJax-Span-75" style="font-size: 70.7%; font-family: MathJax_Math-italic;">y<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.003em;"></span></span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.138em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.434em; border-left: 0px solid; width: 0px; height: 1.003em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>p</mi><mi>y</mi></msub></math></span></span><script type="math/tex" id="MathJax-Element-15">p_y</script>
                </div>
                <div class="clearer"></div>
            </div>
            <div class="slider-control">
                <div class="slider ui-slider ui-slider-horizontal ui-widget ui-widget-content ui-corner-all" id="lookat_pz_slider">
                <a class="ui-slider-handle ui-state-default ui-corner-all" href="#" style="left: 44.6117%;"></a></div>
                <div class="slider-label">
                <span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" id="MathJax-Element-16-Frame" tabindex="0" style="position: relative;" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mi>p</mi><mi>z</mi></msub></math>" role="presentation"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-76" style="width: 1.096em; display: inline-block;"><span style="display: inline-block; position: relative; width: 0.888em; height: 0px; font-size: 120%;"><span style="position: absolute; clip: rect(1.565em, 1000.89em, 2.503em, -999.997em); top: -2.133em; left: 0em;"><span class="mrow" id="MathJax-Span-77"><span class="msubsup" id="MathJax-Span-78"><span style="display: inline-block; position: relative; width: 0.888em; height: 0px;"><span style="position: absolute; clip: rect(3.44em, 1000.52em, 4.378em, -999.997em); top: -4.008em; left: 0em;"><span class="mi" id="MathJax-Span-79" style="font-family: MathJax_Math-italic;">p</span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span><span style="position: absolute; top: -3.852em; left: 0.523em;"><span class="mi" id="MathJax-Span-80" style="font-size: 70.7%; font-family: MathJax_Math-italic;">z<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.003em;"></span></span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.138em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.309em; border-left: 0px solid; width: 0px; height: 0.878em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>p</mi><mi>z</mi></msub></math></span></span><script type="math/tex" id="MathJax-Element-16">p_z</script>
                </div>
                <div class="clearer"></div>
            </div>
            <p>Adjust extrinsic parameters above.</p>

            <p>This is a "look-at" parameterization, which describes the camera's orientation in terms of what it is looking at.  Adjust <span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" id="MathJax-Element-17-Frame" tabindex="0" style="position: relative;" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mi>p</mi><mi>x</mi></msub></math>" role="presentation"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-81" style="width: 1.201em; display: inline-block;"><span style="display: inline-block; position: relative; width: 0.992em; height: 0px; font-size: 120%;"><span style="position: absolute; clip: rect(1.565em, 1000.99em, 2.503em, -999.997em); top: -2.133em; left: 0em;"><span class="mrow" id="MathJax-Span-82"><span class="msubsup" id="MathJax-Span-83"><span style="display: inline-block; position: relative; width: 0.992em; height: 0px;"><span style="position: absolute; clip: rect(3.44em, 1000.52em, 4.378em, -999.997em); top: -4.008em; left: 0em;"><span class="mi" id="MathJax-Span-84" style="font-family: MathJax_Math-italic;">p</span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span><span style="position: absolute; top: -3.852em; left: 0.523em;"><span class="mi" id="MathJax-Span-85" style="font-size: 70.7%; font-family: MathJax_Math-italic;">x</span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.138em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.309em; border-left: 0px solid; width: 0px; height: 0.878em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>p</mi><mi>x</mi></msub></math></span></span><script type="math/tex" id="MathJax-Element-17">p_x</script>, <span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" id="MathJax-Element-18-Frame" tabindex="0" style="position: relative;" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mi>p</mi><mi>y</mi></msub></math>" role="presentation"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-86" style="width: 1.148em; display: inline-block;"><span style="display: inline-block; position: relative; width: 0.94em; height: 0px; font-size: 120%;"><span style="position: absolute; clip: rect(1.565em, 1000.94em, 2.607em, -999.997em); top: -2.133em; left: 0em;"><span class="mrow" id="MathJax-Span-87"><span class="msubsup" id="MathJax-Span-88"><span style="display: inline-block; position: relative; width: 0.94em; height: 0px;"><span style="position: absolute; clip: rect(3.44em, 1000.52em, 4.378em, -999.997em); top: -4.008em; left: 0em;"><span class="mi" id="MathJax-Span-89" style="font-family: MathJax_Math-italic;">p</span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span><span style="position: absolute; top: -3.852em; left: 0.523em;"><span class="mi" id="MathJax-Span-90" style="font-size: 70.7%; font-family: MathJax_Math-italic;">y<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.003em;"></span></span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.138em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.434em; border-left: 0px solid; width: 0px; height: 1.003em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>p</mi><mi>y</mi></msub></math></span></span><script type="math/tex" id="MathJax-Element-18">p_y</script>, and <span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" id="MathJax-Element-19-Frame" tabindex="0" style="position: relative;" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mi>p</mi><mi>z</mi></msub></math>" role="presentation"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-91" style="width: 1.096em; display: inline-block;"><span style="display: inline-block; position: relative; width: 0.888em; height: 0px; font-size: 120%;"><span style="position: absolute; clip: rect(1.565em, 1000.89em, 2.503em, -999.997em); top: -2.133em; left: 0em;"><span class="mrow" id="MathJax-Span-92"><span class="msubsup" id="MathJax-Span-93"><span style="display: inline-block; position: relative; width: 0.888em; height: 0px;"><span style="position: absolute; clip: rect(3.44em, 1000.52em, 4.378em, -999.997em); top: -4.008em; left: 0em;"><span class="mi" id="MathJax-Span-94" style="font-family: MathJax_Math-italic;">p</span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span><span style="position: absolute; top: -3.852em; left: 0.523em;"><span class="mi" id="MathJax-Span-95" style="font-size: 70.7%; font-family: MathJax_Math-italic;">z<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.003em;"></span></span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.138em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.309em; border-left: 0px solid; width: 0px; height: 0.878em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>p</mi><mi>z</mi></msub></math></span></span><script type="math/tex" id="MathJax-Element-19">p_z</script> to change where the camera is looking (orange dot).  The up vector is fixed at (0,1,0)'.  Notice that moving the camera center, *C*, causes the camera to rotate.</p>

            <p>
            </p>
            
        </div>
        <div id="intrinsic-controls" class="ui-tabs-panel ui-widget-content ui-corner-bottom">
            <div class="slider-control">
                <div class="slider ui-slider ui-slider-horizontal ui-widget ui-widget-content ui-corner-all" id="focal_slider">
                <a class="ui-slider-handle ui-state-default ui-corner-all" href="#" style="left: 7.64702%;"></a></div>
                <div class="slider-label">
                Focal Length
                </div>
                <div class="clearer"></div>
            </div>
            <div class="slider-control">
                <div class="slider ui-slider ui-slider-horizontal ui-widget ui-widget-content ui-corner-all" id="skew_slider">
                <a class="ui-slider-handle ui-state-default ui-corner-all" href="#" style="left: 0%;"></a></div>
                <div class="slider-label">
                Axis Skew 
                </div>
                <div class="clearer"></div>
            </div>
            <div class="slider-control">
                <div class="slider ui-slider ui-slider-horizontal ui-widget ui-widget-content ui-corner-all" id="x0_slider">
                <a class="ui-slider-handle ui-state-default ui-corner-all" href="#" style="left: 50%;"></a></div>
                <div class="slider-label">
                <span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" id="MathJax-Element-20-Frame" tabindex="0" style="position: relative;" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mi>x</mi><mn>0</mn></msub></math>" role="presentation"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-96" style="width: 1.201em; display: inline-block;"><span style="display: inline-block; position: relative; width: 0.992em; height: 0px; font-size: 120%;"><span style="position: absolute; clip: rect(1.565em, 1000.99em, 2.451em, -999.997em); top: -2.133em; left: 0em;"><span class="mrow" id="MathJax-Span-97"><span class="msubsup" id="MathJax-Span-98"><span style="display: inline-block; position: relative; width: 0.992em; height: 0px;"><span style="position: absolute; clip: rect(3.44em, 1000.52em, 4.169em, -999.997em); top: -4.008em; left: 0em;"><span class="mi" id="MathJax-Span-99" style="font-family: MathJax_Math-italic;">x</span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span><span style="position: absolute; top: -3.852em; left: 0.576em;"><span class="mn" id="MathJax-Span-100" style="font-size: 70.7%; font-family: MathJax_Main;">0</span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.138em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.247em; border-left: 0px solid; width: 0px; height: 0.878em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>x</mi><mn>0</mn></msub></math></span></span><script type="math/tex" id="MathJax-Element-20">x_0</script>
                </div>
                <div class="clearer"></div>
            </div>
            <div class="slider-control">
                <div class="slider ui-slider ui-slider-horizontal ui-widget ui-widget-content ui-corner-all" id="y0_slider">
                <a class="ui-slider-handle ui-state-default ui-corner-all" href="#" style="left: 49.8%;"></a></div>
                <div class="slider-label">
                <span class="MathJax_Preview" style="color: inherit; display: none;"></span><span class="MathJax" id="MathJax-Element-21-Frame" tabindex="0" style="position: relative;" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mi>y</mi><mn>0</mn></msub></math>" role="presentation"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-101" style="width: 1.148em; display: inline-block;"><span style="display: inline-block; position: relative; width: 0.94em; height: 0px; font-size: 120%;"><span style="position: absolute; clip: rect(1.565em, 1000.94em, 2.503em, -999.997em); top: -2.133em; left: 0em;"><span class="mrow" id="MathJax-Span-102"><span class="msubsup" id="MathJax-Span-103"><span style="display: inline-block; position: relative; width: 0.94em; height: 0px;"><span style="position: absolute; clip: rect(3.44em, 1000.52em, 4.378em, -999.997em); top: -4.008em; left: 0em;"><span class="mi" id="MathJax-Span-104" style="font-family: MathJax_Math-italic;">y<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.003em;"></span></span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span><span style="position: absolute; top: -3.852em; left: 0.471em;"><span class="mn" id="MathJax-Span-105" style="font-size: 70.7%; font-family: MathJax_Main;">0</span><span style="display: inline-block; width: 0px; height: 4.013em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.138em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.309em; border-left: 0px solid; width: 0px; height: 0.878em;"></span></span></nobr><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>y</mi><mn>0</mn></msub></math></span></span><script type="math/tex" id="MathJax-Element-21">y_0</script>
                </div>
                <div class="clearer"></div>
            </div>
            <p>Adjust intrinsic parameters above.  As you adjust these parameters, observe how the viewing volume changes in the left pane:</p>

            <ul>
            <li> Changing the focal length moves the yellow focal plane, which chainges the field-of-view angle of the viewing volume.</li>
            <li> Changing the principal point affects where the green center-line intersects the focal plane.</li>
            <li> Setting skew to non-zero causes the focal plane to be non-rectangular</li>
            </ul>
            
            <p>Intrinsic parameters result in 2D transformations only; the depth of objects are ignored.  To see this, observe how the image in the right pane is affected by changing intrinsic parameters:</p>

            <ul>
                <li>Changing the focal length scales the near sphere and the far sphere equally.</li>
                <li>Changing the principal point has no affect on parallax.</li>
                <li>No combination of intrinsic parameters will reveal occluded parts of an object.</li>
            </ul>
            
        </div>
    </div>
</div>

<br>

[fusion 관련 글 목차](https://gaussian37.github.io/vision-fusion-table/)

<br>
