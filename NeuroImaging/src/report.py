import os, base64
def report(top_image_paths, bottom_image_paths, output_html="report_embedded.html", 
           top_titles=None, bottom_titles=None, 
           partial_leakage=0.0,
           p_leakage_x_min=0.0,
           p_leakage_x_max=0.0,
           p_leakage_x_avg=0.0, 
           p_leakage_y_min=0.0,
           p_leakage_y_max=0.0,
           p_leakage_y_avg=0.0, 
           p_leakage_z_min=0.0,
           p_leakage_z_max=0.0,
           p_leakage_z_avg=0.0,
           
           s_leakage_x_min=0.0,
           s_leakage_x_max=0.0,
           s_leakage_x_avg=0.0, 
           s_leakage_y_min=0.0,
           s_leakage_y_max=0.0,
           s_leakage_y_avg=0.0, 
           s_leakage_z_min=0.0,
           s_leakage_z_max=0.0,
           s_leakage_z_avg=0.0,           
           full_leakage=0.0,
           spatiotemporal=False,
           spatiotemporal_image_path=None):
    def encode_image_base64(path):
        with open(path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")
        ext = os.path.splitext(path)[1][1:]  
        return f"data:image/{ext};base64,{encoded}"

    if top_titles is None:
        top_titles = [f"Top Image {i+1}" for i in range(2)]
    if bottom_titles is None:
        bottom_titles = [f"Bottom Image {i+1}" for i in range(3)]
    try:
        rep_dir = os.path.dirname(os.path.abspath(__file__))
        rep_dir = os.path.join(log_dir, 'report')
    except NameError:
        rep_dir = os.path.join(os.getcwd(), 'report')  

    os.makedirs(rep_dir, exist_ok=True)
    
    top_encoded = [encode_image_base64(p) for p in top_image_paths]
    bottom_encoded = [encode_image_base64(p) for p in bottom_image_paths]

    if spatiotemporal and spatiotemporal_image_path:
        spatiotemporal_encoded = encode_image_base64(spatiotemporal_image_path)
        spatiotemporal_section = f"""
    <h2 style="margin-top: 60px; text-align: center;">SpatioTemporal Analysis</h2>
    <div class="image-row" style="justify-content: center;">
        <div class="image-box">
            <img src="{spatiotemporal_encoded}" alt="SpatioTemporal Analysis" onclick="showModal(this.src)">
            <h3>SpatioTemporal Analysis</h3>
        </div>
    </div>
"""
    else:
        spatiotemporal_section = f"""
    <h2 style="margin-top: 60px; text-align: center; color: #999;">SpatioTemporal Analysis not available</h2>
"""

    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Embedded Correlation Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f9f9f9;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 40px;
        }}
        .top-container {{
            display: flex;
            justify-content: space-between;
            gap: 30px;
            margin-bottom: 60px;
        }}
        .top-row {{
            display: flex;
            justify-content: space-around;
            gap: 30px;
            flex: 2;
        }}
        .top-box {{
            flex: 1;
            text-align: center;
        }}
        .top-box img {{
            width: 100%;
            max-width: 400px;
            height: auto;
            border: 1px solid #ccc;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
            cursor: zoom-in;
        }}
        .top-box h3 {{
            margin-top: 10px;
            font-size: 18px;
            color: #333;
        }}
        .stats-container {{
            flex: 1;
            min-width: 300px;
        }}
        .image-row {{
            display: flex;
            justify-content: space-around;
            align-items: flex-start;
            gap: 20px;
            margin-top: 20px;
        }}
        .image-box {{
            flex: 1;
            text-align: center;
        }}
        .image-box img {{
            width: 100%;
            max-width: 600px;
            height: auto;
            border: 1px solid #ccc;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
            cursor: zoom-in;
        }}
        .image-box h3 {{
            margin-top: 10px;
            font-size: 16px;
            color: #333;
        }}
        /* Stats section */
        .stats {{
            margin-top: 0;
            padding: 15px;
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-family: monospace;
            font-size: 16px;
            line-height: 1.6;
            height: 100%;
            box-sizing: border-box;
        }}
        
        .leakage-details {{
            margin-left: 20px;
            padding-left: 10px;
            border-left: 2px solid #ddd;
        }}

        .stats h3 {{
            color: #333;
            margin: 10px 0;
        }}

        .stats p {{
            margin: 5px 0;
            font-family: monospace;
        }}
        /* Lightbox modal */
        .modal {{
            display: none;
            position: fixed;
            z-index: 10;
            left: 0; top: 0;
            width: 100%; height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.9);
        }}
        .modal-content {{
            margin: auto;
            display: block;
            max-width: 90%;
            max-height: 90%;
        }}
        .close {{
            position: absolute;
            top: 20px; right: 35px;
            color: white;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }}
        @keyframes zoom {{
            from {{transform: scale(0.7)}}
            to {{transform: scale(1)}}
        }}
        .modal-content, .modal {{
            animation: zoom 0.3s;
        }}
    </style>
</head>
<body>
    <h1>Leakage Analysis Report</h1>

    <!-- Top container with images and stats -->
    <div class="top-container">
        <!-- Top side-by-side images -->
        <div class="top-row">
            {"".join(f'''
            <div class="top-box">
                <img src="{top_encoded[i]}" alt="{top_titles[i]}" onclick="showModal(this.src)">
                <h3>{top_titles[i]}</h3>
            </div>''' for i in range(2))}
        </div>

        <!-- Stats section -->
        <div class="stats-container">
            <div class="stats">
                <h3>Result:</h3>
                <p><strong>Full leakage:</strong> {'True' if full_leakage else 'False'}</p>
                <p><strong>Partial leakage:</strong> {partial_leakage:.2f}%</p>
                <p><strong>Pearson Correlation:</strong></p>
                <div class="leakage-details">
                    <p><strong>X</strong>: Min: {p_leakage_x_min:.2f}&emsp;&emsp;&emsp;Max: {p_leakage_x_max:.2f}&emsp;&emsp;&emsp;AVG: {p_leakage_x_avg:.2f}</p>
                    <p><strong>Y</strong>: Min: {p_leakage_y_min:.2f}&emsp;&emsp;&emsp;Max: {p_leakage_y_max:.2f}&emsp;&emsp;&emsp;AVG: {p_leakage_y_avg:.2f}</p>
                    <p><strong>Z</strong>: Min: {p_leakage_z_min:.2f}&emsp;&emsp;&emsp;Max: {p_leakage_z_max:.2f}&emsp;&emsp;&emsp;AVG: {p_leakage_z_avg:.2f}</p>
                </div>
                <p><strong>SSIM Score:</strong></p>
                <div class="leakage-details">
                    <p><strong>X</strong>: Min: {s_leakage_x_min:.2f}&emsp;&emsp;&emsp;Max: {s_leakage_x_max:.2f}&emsp;&emsp;&emsp;AVG: {s_leakage_x_avg:.2f}</p>
                    <p><strong>Y</strong>: Min: {s_leakage_y_min:.2f}&emsp;&emsp;&emsp;Max: {s_leakage_y_max:.2f}&emsp;&emsp;&emsp;AVG: {s_leakage_y_avg:.2f}</p>
                    <p><strong>Z</strong>: Min: {s_leakage_z_min:.2f}&emsp;&emsp;&emsp;Max: {s_leakage_z_max:.2f}&emsp;&emsp;&emsp;AVG: {s_leakage_z_avg:.2f}</p>
                </div>
            </div>
        </div>
    </div>

    <!-- Spatial Analysis section -->
    <h2 style="margin-top: 60px; text-align: center;">Spatial Analysis</h2>
    <div class="image-row">
        {"".join(f'''
        <div class="image-box">
            <img src="{bottom_encoded[i]}" alt="{bottom_titles[i]}" onclick="showModal(this.src)">
            <h3>{bottom_titles[i]}</h3>
        </div>''' for i in range(3))}
    </div>

    <!-- Spatiotemporal Analysis section (conditional) -->
    {spatiotemporal_section}

    <!-- Lightbox modal -->
    <div id="imgModal" class="modal" onclick="closeModal()">
        <span class="close">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>

    <script>
        function showModal(src) {{
            const modal = document.getElementById("imgModal");
            const modalImg = document.getElementById("modalImage");
            modal.style.display = "block";
            modalImg.src = src;
        }}
        function closeModal() {{
            document.getElementById("imgModal").style.display = "none";
        }}
    </script>
</body>
</html>
"""
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_template)
