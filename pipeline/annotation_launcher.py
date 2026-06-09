"""GUI launcher for top-level integrated annotation inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
FIXED_SNAP_CONFIG_PATH = "interactive_snapping_2d/config/snap_config.json"
LAUNCHER_GEOMETRY = "960x720"
PREVIEW_SIZE = (520, 300)


@dataclass(frozen=True)
class SAM2ModelOption:
    """One fixed SAM2 model option exposed by the launcher."""

    key: str
    label: str
    description: str
    checkpoint: str
    model_cfg: str


@dataclass(frozen=True)
class AnnotationLauncherSelection:
    """Resolved input paths selected by the annotation launcher."""

    image: str
    workspace: str
    output_annotation: str
    sam2_checkpoint: str
    sam2_model_cfg: str


@dataclass(frozen=True)
class AnnotationLauncherDefaults:
    """Initial values for the annotation launcher."""

    image: str
    workspace: str
    output_annotation: str
    data_dir: str
    model_key: str


SAM2_MODEL_OPTIONS = (
    SAM2ModelOption(
        key="tiny",
        label="Tiny",
        description="Fastest option with the lowest memory use; best for quick tests.",
        checkpoint="sam2/checkpoints/sam2.1_hiera_tiny.pt",
        model_cfg="configs/sam2.1/sam2.1_hiera_t.yaml",
    ),
    SAM2ModelOption(
        key="small",
        label="Small",
        description="Fast model with moderate memory use; useful for lightweight annotation.",
        checkpoint="sam2/checkpoints/sam2.1_hiera_small.pt",
        model_cfg="configs/sam2.1/sam2.1_hiera_s.yaml",
    ),
    SAM2ModelOption(
        key="base_plus",
        label="Base Plus",
        description="Balanced quality and speed for regular annotation work.",
        checkpoint="sam2/checkpoints/sam2.1_hiera_base_plus.pt",
        model_cfg="configs/sam2.1/sam2.1_hiera_b+.yaml",
    ),
    SAM2ModelOption(
        key="large",
        label="Large",
        description="Best mask quality with the highest memory use.",
        checkpoint="sam2/checkpoints/sam2.1_hiera_large.pt",
        model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
    ),
)


def repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[1]


def default_data_dir() -> Path:
    """Return the default image browse directory."""
    data_dir = repo_root() / "data"
    return data_dir if data_dir.exists() else repo_root()


def default_annotation_root() -> Path:
    """Return the default annotation workspace root."""
    return repo_root() / "annotation"


def model_option_by_key(key: str) -> SAM2ModelOption:
    """Return a fixed SAM2 model option by key."""
    for option in SAM2_MODEL_OPTIONS:
        if option.key == key:
            return option
    raise KeyError(f"Unknown SAM2 model option: {key}")


def is_model_available(option: SAM2ModelOption) -> bool:
    """Return whether the model checkpoint exists locally."""
    return (repo_root() / option.checkpoint).is_file()


def default_model_key() -> str:
    """Return the default SAM2 model key."""
    for key in ("large", "base_plus", "small", "tiny"):
        if is_model_available(model_option_by_key(key)):
            return key
    return "large"


def is_supported_image_path(path: str | Path) -> bool:
    """Return whether a path has a supported image extension."""
    return Path(path).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def list_image_browser_entries(folder: str | Path) -> tuple[list[Path], list[Path]]:
    """Return subdirectories and supported images for the launcher file list."""
    folder_path = Path(folder).expanduser()
    if not folder_path.is_dir():
        return [], []
    directories: list[Path] = []
    images: list[Path] = []
    for child in sorted(folder_path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        if child.is_dir():
            directories.append(child)
        elif child.is_file() and is_supported_image_path(child):
            images.append(child)
    return directories, images


def default_workspace_for_image(image_path: str | Path | None) -> Path:
    """Return the default workspace for an input image."""
    if image_path is None:
        name = "annotation"
    else:
        path = Path(image_path).expanduser()
        name = path.parent.name or path.stem or "annotation"
    return default_annotation_root() / name


def output_annotation_for_workspace(workspace: str | Path, parser_output_annotation: str | Path | None = None) -> Path:
    """Return the final annotation JSON path for a selected workspace."""
    workspace_path = Path(workspace).expanduser()
    if parser_output_annotation is None:
        return workspace_path / "annotation.json"
    parser_output_path = Path(parser_output_annotation).expanduser()
    if parser_output_path.parent.resolve() == workspace_path.resolve():
        return parser_output_path
    return workspace_path / "annotation.json"


def _path_or_empty(path: str | Path | None) -> str:
    if path is None:
        return ""
    return str(Path(path).expanduser())


def _initial_data_dir(image: str | Path | None) -> Path:
    if image is not None:
        path = Path(image).expanduser()
        if path.is_file():
            return path.parent
        if path.is_dir():
            return path
    return default_data_dir()


def build_launcher_defaults(
    image: str | Path | None = None,
    workspace: str | Path | None = None,
    output_annotation: str | Path | None = None,
    model_key: str | None = None,
) -> AnnotationLauncherDefaults:
    """Build launcher defaults from optional parser-provided values."""
    image_value = _path_or_empty(image)
    if workspace is not None:
        workspace_path = Path(workspace).expanduser()
    elif output_annotation is not None:
        workspace_path = Path(output_annotation).expanduser().parent
    else:
        workspace_path = default_workspace_for_image(image)

    output_path = Path(output_annotation).expanduser() if output_annotation is not None else workspace_path / "annotation.json"
    return AnnotationLauncherDefaults(
        image=image_value,
        workspace=str(workspace_path),
        output_annotation=str(output_path),
        data_dir=str(_initial_data_dir(image)),
        model_key=model_key or default_model_key(),
    )


def launch_annotation_inputs(
    image: str | Path | None = None,
    workspace: str | Path | None = None,
    output_annotation: str | Path | None = None,
) -> AnnotationLauncherSelection | None:
    """Open the annotation launcher and return selected paths, or ``None`` when cancelled."""
    import tkinter as tk
    from PIL import Image, ImageTk

    defaults = build_launcher_defaults(image, workspace, output_annotation)
    root = tk.Tk()
    root.title("Integrated Annotation Setup")
    root.geometry(LAUNCHER_GEOMETRY)
    root.minsize(960, 720)
    root.maxsize(960, 720)
    root.resizable(False, False)

    result: dict[str, AnnotationLauncherSelection | None] = {"selection": None}
    current_step = {"value": 0}
    current_folder = {"path": Path(defaults.data_dir).expanduser()}
    file_entries: list[Path] = []
    selected_image = {"path": Path(defaults.image).expanduser() if defaults.image else None}
    preview_ref: dict[str, object | None] = {"image": None}
    output_manual = {"value": workspace is not None or output_annotation is not None}

    folder_var = tk.StringVar(value=str(current_folder["path"]))
    selected_image_var = tk.StringVar(value=str(selected_image["path"] or ""))
    workspace_var = tk.StringVar(value=defaults.workspace)
    output_annotation_var = tk.StringVar(value=defaults.output_annotation)
    model_var = tk.StringVar(value=defaults.model_key)
    status_var = tk.StringVar(value="Select an image from the data folder.")

    main = tk.Frame(root, padx=16, pady=14)
    main.pack(fill="both", expand=True)
    header = tk.Frame(main, height=48)
    header.pack(fill="x")
    step_label = tk.Label(header, text="", anchor="w", font=("TkDefaultFont", 13, "bold"))
    step_label.pack(side="left")
    status_label = tk.Label(main, textvariable=status_var, anchor="w", fg="#444444")
    status_label.pack(fill="x", pady=(0, 8))
    content = tk.Frame(main, width=928, height=560, relief="groove", borderwidth=1)
    content.pack(fill="both")
    content.pack_propagate(False)
    footer = tk.Frame(main, height=56)
    footer.pack(fill="x", pady=(12, 0))

    back_button = tk.Button(footer, text="Back", width=10)
    next_button = tk.Button(footer, text="Next", width=10)
    cancel_button = tk.Button(footer, text="Cancel", width=10)
    start_button = tk.Button(footer, text="Start", width=10)
    cancel_button.pack(side="right", padx=(8, 0))
    start_button.pack(side="right", padx=(8, 0))
    next_button.pack(side="right", padx=(8, 0))
    back_button.pack(side="right")

    def clear_content() -> None:
        for child in content.winfo_children():
            child.destroy()

    def set_status(message: str) -> None:
        status_var.set(message)

    def sync_workspace_from_image() -> None:
        image_path = selected_image["path"]
        if image_path is None or output_manual["value"]:
            return
        workspace_path = default_workspace_for_image(image_path)
        workspace_var.set(str(workspace_path))
        output_annotation_var.set(str(output_annotation_for_workspace(workspace_path, output_annotation)))

    def refresh_output_annotation(*_args: object) -> None:
        workspace_value = workspace_var.get().strip()
        if not workspace_value:
            output_annotation_var.set("")
            return
        output_annotation_var.set(str(output_annotation_for_workspace(workspace_value, output_annotation)))

    workspace_var.trace_add("write", refresh_output_annotation)

    def render_preview(path: Path | None) -> None:
        preview_label = getattr(root, "_preview_label", None)
        if preview_label is None:
            return
        if path is None or not path.is_file():
            preview_ref["image"] = None
            preview_label.configure(image="", text="No image selected")
            return
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail(PREVIEW_SIZE)
                photo = ImageTk.PhotoImage(img)
        except OSError as exc:
            preview_ref["image"] = None
            preview_label.configure(image="", text=f"Preview unavailable: {exc}")
            return
        preview_ref["image"] = photo
        preview_label.configure(image=photo, text="")

    def load_folder(folder: Path) -> None:
        nonlocal file_entries
        folder = folder.expanduser()
        if not folder.is_dir():
            set_status(f"Folder does not exist: {folder}")
            return
        current_folder["path"] = folder
        folder_var.set(str(folder))
        directories, images = list_image_browser_entries(folder)
        file_entries = directories + images
        listbox = getattr(root, "_image_listbox", None)
        if listbox is None:
            return
        listbox.delete(0, tk.END)
        for item in file_entries:
            prefix = "[DIR] " if item.is_dir() else ""
            listbox.insert(tk.END, f"{prefix}{item.name}")
        set_status(f"{len(images)} image(s) found in {folder}.")

    def select_image(path: Path) -> None:
        selected_image["path"] = path
        selected_image_var.set(str(path))
        sync_workspace_from_image()
        render_preview(path)
        set_status(f"Selected image: {path.name}")

    def on_file_select(_event: object | None = None) -> None:
        listbox = getattr(root, "_image_listbox", None)
        if listbox is None:
            return
        selection = listbox.curselection()
        if not selection:
            return
        path = file_entries[int(selection[0])]
        if path.is_file():
            select_image(path)

    def on_file_open(_event: object | None = None) -> None:
        listbox = getattr(root, "_image_listbox", None)
        if listbox is None:
            return
        selection = listbox.curselection()
        if not selection:
            return
        path = file_entries[int(selection[0])]
        if path.is_dir():
            load_folder(path)
        elif path.is_file():
            select_image(path)

    def go_parent() -> None:
        load_folder(current_folder["path"].parent)

    def load_typed_folder() -> None:
        load_folder(Path(folder_var.get().strip()))

    def show_image_step() -> None:
        clear_content()
        step_label.configure(text="Step 1 of 3: Select Input Image")
        top = tk.Frame(content, padx=12, pady=10)
        top.pack(fill="x")
        tk.Label(top, text="Folder").pack(side="left")
        tk.Entry(top, textvariable=folder_var, width=74).pack(side="left", padx=8)
        tk.Button(top, text="Open", command=load_typed_folder, width=8).pack(side="left", padx=(0, 6))
        tk.Button(top, text="Up", command=go_parent, width=8).pack(side="left")

        body = tk.Frame(content, padx=12)
        body.pack(fill="both", expand=True)
        list_frame = tk.Frame(body, width=350, height=450)
        list_frame.pack(side="left", fill="y")
        list_frame.pack_propagate(False)
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        image_listbox = tk.Listbox(list_frame, width=42, height=25, exportselection=False, yscrollcommand=scrollbar.set)
        image_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.configure(command=image_listbox.yview)
        image_listbox.bind("<<ListboxSelect>>", on_file_select)
        image_listbox.bind("<Double-Button-1>", on_file_open)
        root._image_listbox = image_listbox

        preview_frame = tk.Frame(body, width=540, height=450)
        preview_frame.pack(side="left", fill="both", expand=True, padx=(16, 0))
        preview_frame.pack_propagate(False)
        tk.Label(preview_frame, text="Selected image").pack(anchor="w")
        tk.Label(preview_frame, textvariable=selected_image_var, anchor="w", wraplength=520).pack(fill="x", pady=(2, 8))
        preview_box = tk.Frame(preview_frame, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1], bg="#eeeeee", relief="sunken", borderwidth=1)
        preview_box.pack(anchor="w")
        preview_box.pack_propagate(False)
        preview_label = tk.Label(preview_box, text="No image selected", bg="#eeeeee")
        preview_label.pack(fill="both", expand=True)
        root._preview_label = preview_label
        load_folder(current_folder["path"])
        render_preview(selected_image["path"])

    def show_output_step() -> None:
        clear_content()
        step_label.configure(text="Step 2 of 3: Select Output Directory")
        frame = tk.Frame(content, padx=24, pady=24)
        frame.pack(fill="both", expand=True)
        image_path = selected_image["path"]
        if image_path is not None and not output_manual["value"]:
            sync_workspace_from_image()
        tk.Label(frame, text="Input image").grid(row=0, column=0, sticky="w", pady=(0, 8))
        tk.Label(frame, text=str(image_path or ""), anchor="w", wraplength=720).grid(row=0, column=1, sticky="ew", pady=(0, 8))
        tk.Label(frame, text="Output directory").grid(row=1, column=0, sticky="w", pady=(8, 8))
        workspace_entry = tk.Entry(frame, textvariable=workspace_var, width=88)
        workspace_entry.grid(row=1, column=1, sticky="ew", pady=(8, 8))
        tk.Label(frame, text="Final annotation JSON").grid(row=2, column=0, sticky="w", pady=(8, 8))
        tk.Label(frame, textvariable=output_annotation_var, anchor="w", wraplength=720).grid(row=2, column=1, sticky="ew", pady=(8, 8))
        tk.Label(
            frame,
            text="The output directory is also the annotation workspace. It will be created when Start is pressed.",
            anchor="w",
            wraplength=780,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(20, 0))
        frame.columnconfigure(1, weight=1)

        def mark_manual(_event: object | None = None) -> None:
            output_manual["value"] = True
            refresh_output_annotation()

        workspace_entry.bind("<KeyRelease>", mark_manual)
        workspace_entry.focus_set()
        set_status("Confirm or edit the output directory.")

    def show_model_step() -> None:
        clear_content()
        step_label.configure(text="Step 3 of 3: Select SAM2 Model")
        frame = tk.Frame(content, padx=24, pady=18)
        frame.pack(fill="both", expand=True)
        tk.Label(frame, text="Choose one SAM2.1 model. Missing checkpoints cannot be selected.", anchor="w").pack(fill="x", pady=(0, 12))
        for option in SAM2_MODEL_OPTIONS:
            available = is_model_available(option)
            row = tk.Frame(frame, height=86, relief="groove", borderwidth=1, padx=10, pady=8)
            row.pack(fill="x", pady=6)
            row.pack_propagate(False)
            state = "normal" if available else "disabled"
            label = option.label if available else f"{option.label} (missing checkpoint)"
            tk.Radiobutton(row, text=label, variable=model_var, value=option.key, state=state).pack(anchor="w")
            tk.Label(row, text=option.description, anchor="w", fg="#444444").pack(anchor="w", pady=(4, 0))
        set_status("Select the SAM2 model for ROI annotation.")

    def can_leave_current_step() -> bool:
        if current_step["value"] == 0:
            image_path = selected_image["path"]
            if image_path is None or not image_path.is_file():
                set_status("Select an input image before continuing.")
                return False
        if current_step["value"] == 1:
            if not workspace_var.get().strip():
                set_status("Enter an output directory before continuing.")
                return False
        return True

    def show_current_step() -> None:
        back_button.configure(state="normal" if current_step["value"] > 0 else "disabled")
        next_button.configure(state="normal" if current_step["value"] < 2 else "disabled")
        start_button.configure(state="normal" if current_step["value"] == 2 else "disabled")
        if current_step["value"] == 0:
            show_image_step()
        elif current_step["value"] == 1:
            show_output_step()
        else:
            show_model_step()

    def go_next() -> None:
        if not can_leave_current_step():
            return
        current_step["value"] = min(2, current_step["value"] + 1)
        show_current_step()

    def go_back() -> None:
        current_step["value"] = max(0, current_step["value"] - 1)
        show_current_step()

    def cancel() -> None:
        result["selection"] = None
        root.destroy()

    def start() -> None:
        if not can_leave_current_step():
            return
        image_path = selected_image["path"]
        if image_path is None or not image_path.is_file():
            set_status("Select an input image before starting.")
            return
        workspace_value = workspace_var.get().strip()
        if not workspace_value:
            set_status("Enter an output directory before starting.")
            return
        model_option = model_option_by_key(model_var.get())
        checkpoint_path = repo_root() / model_option.checkpoint
        if not checkpoint_path.is_file():
            set_status(f"Missing checkpoint for {model_option.label}: {checkpoint_path}")
            return
        workspace_path = Path(workspace_value).expanduser()
        try:
            workspace_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            set_status(f"Cannot create output directory: {exc}")
            return
        annotation_path = output_annotation_for_workspace(workspace_path, output_annotation)
        result["selection"] = AnnotationLauncherSelection(
            image=str(image_path.resolve()),
            workspace=str(workspace_path.resolve()),
            output_annotation=str(annotation_path.resolve()),
            sam2_checkpoint=str(checkpoint_path.resolve()),
            sam2_model_cfg=model_option.model_cfg,
        )
        root.destroy()

    back_button.configure(command=go_back)
    next_button.configure(command=go_next)
    cancel_button.configure(command=cancel)
    start_button.configure(command=start)
    root.protocol("WM_DELETE_WINDOW", cancel)

    if selected_image["path"] is not None and selected_image["path"].is_file():
        sync_workspace_from_image()
    show_current_step()
    root.mainloop()
    return result["selection"]
