"""
To use this config file, symlink it to ~/.jupyter/jupyter_notebook_config.py
"""

import io
import os
from notebook.utils import to_api_path

_script_exporter = None
_html_exporter = None


def save_as_html(os_path, contents_manager):
    from nbconvert.exporters.html import HTMLExporter

    global _html_exporter

    if _html_exporter is None:
        _html_exporter = HTMLExporter(parent=contents_manager)

    log = contents_manager.log
    base, ext = os.path.splitext(os_path)
    html_fname = base + '.html'
    html, resources = _html_exporter.from_filename(os_path)

    if resources.get('output_extension') == '.txt':
        # prevent annoying behaviour of also saving as file.txt
        return
    
    html_fname = base + resources.get('output_extension', '.txt')
    log.info("Saving HTML /%s", to_api_path(html_fname, contents_manager.root_dir))

    with io.open(html_fname, 'w', encoding='utf-8') as f:
        f.write(html)

        
def save_as_python(os_path, contents_manager):
    from nbconvert.exporters.script import ScriptExporter

    global _script_exporter

    if _script_exporter is None:
        _script_exporter = ScriptExporter(parent=contents_manager)

    log = contents_manager.log
    base, ext = os.path.splitext(os_path)
    py_fname = base + '.py'
    script, resources = _script_exporter.from_filename(os_path)

    if resources.get('output_extension') == '.txt':
        # prevent annoying behaviour of also saving as file.txt
        return

    script_fname = base + resources.get('output_extension', '.txt')
    log.info("Saving script /%s", to_api_path(script_fname, contents_manager.root_dir))

    with io.open(script_fname, 'w', encoding='utf-8') as f:
        f.write(script)
    

def script_post_save(model, os_path, contents_manager, **kwargs):
    if model['type'] != 'notebook':
        return

    save_as_python(os_path, contents_manager)
    save_as_html(os_path, contents_manager)

    
c.FileContentsManager.post_save_hook = script_post_save
