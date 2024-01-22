Make a venv (change path if you are serious)

```bash
python3 -m venv yolo ./env
```

Activate it:

```bash
source ./env/bin/activate
```

Install the dependencies (pray to god everything goes well)

```
pip3 install -r requirements.txt
```

# Showtime

The DeepSORT tracker follows the upper-left corner of the bounding box identified by YOLOv8. When this corner crosses a predefined zone, marked by subtly visible lines, it registers an entry, effectively counting people entering that area.
![](test0.png)
