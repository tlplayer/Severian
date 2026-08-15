# Compile-time trait registries

`File` and `Image` stay generic while reachable packages contribute concrete
providers. `FileType` and `ImageType` are typed identifiers, not open enums, so
adding a dependency never changes exhaustiveness for an existing enum.

## Four file providers

```sev
class FileType:
    name: string

class FileData:
    bytes: list[int]

trait File:
    @file
    property file_type: FileType
    property extensions: set[string]
    def read(path: string) -> FileData

class JsonFile: File
    file_type = FileType("json")
    extensions = {".json"}

    def read(path: string) -> FileData:
        return json.parse(bytes.read(path))

class TextFile: File
    file_type = FileType("text")
    extensions = {".txt"}

    def read(path: string) -> FileData:
        return text.decode(bytes.read(path))

class YamlFile: File
    file_type = FileType("yaml")
    extensions = {".yaml", ".yml"}

    def read(path: string) -> FileData:
        return yaml.parse(bytes.read(path))

class LuaFile: File
    file_type = FileType("lua")
    extensions = {".lua"}

    def read(path: string) -> FileData:
        return lua.parse(bytes.read(path))
```

The closed registry is equivalent to:

| Lookup key | Provider |
| --- | --- |
| `.json` | `JsonFile` |
| `.txt` | `TextFile` |
| `.yaml`, `.yml` | `YamlFile` |
| `.lua` | `LuaFile` |

The intended generic surface remains:

```sev
config = file.read("config.lua")
settings = file.read("settings.json")
notes = file.read("notes.txt")
pipeline = file.read("pipeline.yml")
```

## Four image providers

```sev
class ImageType:
    name: string

class ImageData:
    pixels: list[int]

trait Image:
    @image
    property image_type: ImageType
    property extensions: set[string]
    def read(path: string) -> ImageData

class PngImage: Image
    image_type = ImageType("png")
    extensions = {".png"}

    def read(path: string) -> ImageData:
        return png.decode(bytes.read(path))

class JpegImage: Image
    image_type = ImageType("jpeg")
    extensions = {".jpg", ".jpeg"}

    def read(path: string) -> ImageData:
        return jpeg.decode(bytes.read(path))

class WebpImage: Image
    image_type = ImageType("webp")
    extensions = {".webp"}

    def read(path: string) -> ImageData:
        return webp.decode(bytes.read(path))

class GifImage: Image
    image_type = ImageType("gif")
    extensions = {".gif"}

    def read(path: string) -> ImageData:
        return gif.decode(bytes.read(path))
```

The compiler sees:

| Lookup key | Provider |
| --- | --- |
| `.png` | `PngImage` |
| `.jpg`, `.jpeg` | `JpegImage` |
| `.webp` | `WebpImage` |
| `.gif` | `GifImage` |

```sev
logo = image.read("logo.png")
photo = image.read("photo.jpg")
thumbnail = image.read("thumbnail.webp")
animation = image.read("loading.gif")
```

Packages do not call `register` and do not run initialization hooks. If a Lua,
WebP, or GIF package is reachable, its implementation is present in the static
registry. If it is absent, so is the provider. Duplicate lookup values are a
compile-time ambiguity instead of an import-order decision.
