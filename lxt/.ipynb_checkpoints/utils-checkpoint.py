import matplotlib.cm as cm
import matplotlib.colors as colors
import os
import subprocess
from pathlib import Path


# save gradient info 

def save_grad_info(model, save_path='weight_gradients.json'):
    # 获取权重名称和对应的梯度
    weight_gradients = {}

    # 遍历模型的所有命名参数
    for name, param in model.named_parameters():
        # 检查参数是否有梯度
        if param.grad is not None:
            # 存储权重名称和对应的梯度
            weight_gradients[name] = param.grad.clone().detach()
            print(f"Parameter: {name}")
            print(f"  Gradient shape: {param.grad.shape}")
            print(f"  Gradient norm: {torch.norm(param.grad):.6f}")
            print(f"  Gradient stats - min: {param.grad.min():.6f}, max: {param.grad.max():.6f}, mean: {param.grad.mean():.6f}")
            print("-" * 80)

    # 可选：按梯度范数排序
    print("\n=== 按梯度范数排序 ===")
    sorted_gradients = sorted(weight_gradients.items(), key=lambda x: torch.norm(x[1]), reverse=True)

    for name, grad in sorted_gradients[:10]:  # 显示前10个梯度最大的权重
        print(f"{name}: norm = {torch.norm(grad):.6f}")

    # 可选：保存梯度信息到文件
    import json
    import numpy as np

    # 将梯度信息转换为可序列化的格式
    gradient_info = {}
    for name, grad in weight_gradients.items():
        gradient_info[name] = {
            'shape': list(grad.shape),
            'norm': float(torch.norm(grad).cpu().numpy()),
            'mean': float(grad.mean().cpu().numpy()),
            'std': float(grad.std().cpu().numpy()),
            'min': float(grad.min().cpu().numpy()),
            'max': float(grad.max().cpu().numpy())
        }

    # 保存到JSON文件
    with open(save_path, 'w') as f:
        json.dump(gradient_info, f, indent=2)

    print(f"\n梯度信息已保存到 weight_gradients.json，共 {len(weight_gradients)} 个参数")

def _apply_colormap(relevance, cmap):
    
    colormap = cm.get_cmap(cmap)
    return colormap(colors.Normalize(vmin=-1, vmax=1)(relevance))

def _generate_latex_org(words, relevances, cmap="bwr"):
    """
    Generate LaTeX code for a sentence with colored words based on their relevances.
    """

    # Generate LaTeX code
    latex_code = r'''
    \documentclass[arwidth=200mm]{standalone} 
    \usepackage[dvipsnames]{xcolor}
    
    \begin{document}
    \fbox{
    \parbox{\textwidth}{
    \setlength\fboxsep{0pt}
    '''

    for word, relevance in zip(words, relevances):
        rgb = _apply_colormap(relevance, cmap)
        r, g, b = int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)

        if word.startswith(' '):
            latex_code += f' \\colorbox[RGB]{{{r},{g},{b}}}{{\\strut {word}}}'
        else:
            latex_code += f'\\colorbox[RGB]{{{r},{g},{b}}}{{\\strut {word}}}'


    latex_code += r'}}\end{document}'

    return latex_code



import os
import re
import subprocess
import tempfile
from pathlib import Path

def _generate_latex(content: str, output_pdf: str):
    """
    生成 PDF：若检测到中文，则使用 XeLaTeX + xeCJK；否则使用 pdflatex。
    """
    # 自动检测是否包含中文 / 日文 / 韩文字符
    contains_cjk = bool(re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', content))

    # LaTeX 模板
    if contains_cjk:
        # 中文模式：使用 XeLaTeX + Noto Sans CJK 字体
        latex_template = r"""
\documentclass[12pt]{article}
\usepackage[UTF8]{inputenc}
\usepackage{xeCJK}
\setCJKmainfont{Noto Sans CJK SC}
\usepackage{geometry}
\geometry{a4paper,margin=1in}
\usepackage{hyperref}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\linespread{1.3}
\begin{document}
%s
\end{document}
""" % content
    else:
        # 英文模式：普通 pdflatex 模板
        latex_template = r"""
\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\geometry{a4paper,margin=1in}
\usepackage{hyperref}
\linespread{1.3}
\begin{document}
%s
\end{document}
""" % content

    # 写入临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_file = Path(tmpdir) / "doc.tex"
        tex_file.write_text(latex_template, encoding="utf-8")

        compiler = "xelatex" if contains_cjk else "pdflatex"
        cmd = [compiler, "-interaction=nonstopmode", str(tex_file)]
        print(f"[INFO] Using {compiler} for PDF generation...")

        subprocess.run(cmd, cwd=tmpdir, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        pdf_path = Path(tmpdir) / "doc.pdf"
        if pdf_path.exists():
            os.rename(pdf_path, output_pdf)
            print(f"[SUCCESS] PDF saved to: {output_pdf}")
        else:
            raise RuntimeError("PDF generation failed. Check LaTeX log for details.")




    
def _compile_latex_to_pdf(latex_code, path='word_colors.pdf', delete_aux_files=True, backend='xelatex'):
    """
    Compile LaTeX code to a PDF file using pdflatex or xelatex.
    """
    
    # Save LaTeX code to a file
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)

    with open(path.with_suffix(".tex"), 'w') as f:
        f.write(latex_code)

    # Use pdflatex to generate PDF file
    if backend == 'pdflatex':
        subprocess.call(['pdflatex', '--output-directory', path.parent, path.with_suffix(".tex")])
    elif backend == 'xelatex':
        subprocess.call(['xelatex', '--output-directory', path.parent, path.with_suffix(".tex")])

    print("PDF file generated successfully.")

    if delete_aux_files:
        for suffix in ['.aux', '.log', '.tex']:
            os.remove(path.with_suffix(suffix))


def pdf_heatmap(words, relevances, cmap="bwr", path='heatmap.pdf', delete_aux_files=True, backend='xelatex'):
    """
    Generate a PDF file with a heatmap of the relevances of the words in a sentence using LaTeX.

    Parameters
    ----------
    words : list of str
        The words in the sentence.
    relevances : list of float
        The relevances of the words normalized between -1 and 1.
    cmap : str
        The name of the colormap to use.
    path : str
        The path to save the PDF file.
    delete_aux_files : bool
        Whether to delete the auxiliary files generated by LaTeX.
    backend : str
        The LaTeX backend to use (pdflatex or xelatex).
    """

    assert len(words) == len(relevances), "The number of words and relevances must be the same."
    assert relevances.min() >= -1 and relevances.max() <= 1, "The relevances must be normalized between -1 and 1."

    latex_code = _generate_latex(words, relevances, cmap=cmap)
    _compile_latex_to_pdf(latex_code, path=path, delete_aux_files=delete_aux_files, backend=backend)


def clean_tokens(words):
    """
    Clean wordpiece tokens by removing special characters and splitting them into words.
    """

    if any("▁" in word for word in words):
        words = [word.replace("▁", " ") for word in words]
    
    elif any("Ġ" in word for word in words):
        words = [word.replace("Ġ", " ") for word in words]
    
    elif any("##" in word for word in words):
        words = [word.replace("##", "") if "##" in word else " " + word for word in words]
        words[0] = words[0].strip()

    else:
        raise ValueError("The tokenization scheme is not recognized.")
    
    special_characters = ['&', '%', '$', '#', '_', '{', '}', '\\']
    for i, word in enumerate(words):
        for special_character in special_characters:
            if special_character in word:
                words[i] = word.replace(special_character, '\\' + special_character)

    return words

