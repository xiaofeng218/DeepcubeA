# rubiks_color_change.py
# 运行: python rubiks_color_change.py
# 会生成 rubiks_color_change.html

import json
import model.Cube as Cube

def generate_rubiks_html(initial_state, indince, moves, output_file="rubiks_move.html"):
    """
    initial_state: dict, {face_name: [9个颜色字符串]}
        face_name: U, D, F, B, L, R
        颜色字符串可用 "white","yellow","red","orange","blue","green"
    moves: list, 每步是一个新的状态 (即9个颜色变化后的完整魔方)
    """

    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Rubik's Cube Color Change Animation</title>
<style>
  body {{ margin: 0; background: #000; }}
  canvas {{ display: block; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/three@0.148.0/build/three.min.js"></script>
<script>
  // OrbitControls implementation
  THREE.OrbitControls = function (camera, domElement) {{
    this.camera = camera;
    this.domElement = domElement || document;
    this.enabled = true;
    this.target = new THREE.Vector3();
    this.minDistance = 0;
    this.maxDistance = Infinity;
    this.minPolarAngle = 0;
    this.maxPolarAngle = Math.PI;
    this.minAzimuthAngle = -Infinity;
    this.maxAzimuthAngle = Infinity;
    this.dampingFactor = 0.05;
    this.enableDamping = false;
    this.enableZoom = true;
    this.zoomSpeed = 0.1;
    this.enableRotate = true;
    this.rotateSpeed = 0.1;
    this.enablePan = true;
    this.panSpeed = 0.1;
    this.screenSpacePanning = true;
    this.keyPanSpeed = 0.5;
    this.autoRotate = false;
    this.autoRotateSpeed = 0.2;
    this.keys = {{
      LEFT: 37,
      UP: 38,
      RIGHT: 39,
      BOTTOM: 40
    }};
    this.mouseButtons = {{
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.PAN
    }};
    var scope = this;
    var EPS = 0.000001;
    var spherical = new THREE.Spherical();
    var sphericalDelta = new THREE.Spherical();
    var scale = 1;
    var panOffset = new THREE.Vector3();
    var lastPosition = new THREE.Vector3();
    var state = {{
      NONE: -1,
      ROTATE: 0,
      DOLLY: 1,
      PAN: 2,
      TOUCH_ROTATE: 3,
      TOUCH_DOLLY_PAN: 4
    }};
    var currentState = state.NONE;
    var touchStartTime = 0;
    var touchStartPosition = new THREE.Vector2();
    var touchStartDistance = 0;
    var mouseDownPosition = new THREE.Vector2();
    function getPolarAngle() {{
      return Math.atan2(Math.sqrt(camera.position.x * camera.position.x + camera.position.z * camera.position.z), camera.position.y);
    }}
    function getAzimuthalAngle() {{
      return Math.atan2(camera.position.x, camera.position.z);
    }}
    function rotateLeft(angle) {{
      sphericalDelta.phi -= angle;
    }}
    function rotateUp(angle) {{
      sphericalDelta.theta -= angle;
    }}
    function panLeft(distance, objectMatrix) {{
      var v = new THREE.Vector3();
      v.setFromMatrixColumn(objectMatrix, 0);
      v.multiplyScalar(-distance);
      panOffset.add(v);
    }}
    function panUp(distance, objectMatrix) {{
      var v = new THREE.Vector3();
      if (scope.screenSpacePanning === true) {{
        v.setFromMatrixColumn(objectMatrix, 1);
      }} else {{
        v.setFromMatrixColumn(objectMatrix, 0);
        v.crossVectors(scope.camera.up, v);
      }}
      v.multiplyScalar(distance);
      panOffset.add(v);
    }}
    function pan(deltaX, deltaY) {{
      var element = scope.domElement === document ? scope.domElement.body : scope.domElement;
      if (scope.screenSpacePanning === true) {{
        var position = new THREE.Vector3();
        position.setFromMatrixPosition(scope.camera.matrixWorld);
        var target = new THREE.Vector3();
        target.copy(scope.target);
        var dir = new THREE.Vector3();
        dir.subVectors(position, target);
        var targetDistance = dir.length();
        dir.normalize();
        var eye = new THREE.Vector3();
        eye.copy(position);
        var end = new THREE.Vector3();
        end.copy(target);
        var tempTarget = new THREE.Vector3();
        tempTarget.copy(target);
        var matrix = new THREE.Matrix4();
        matrix.lookAt(eye, end, scope.camera.up);
        var right = new THREE.Vector3();
        right.setFromMatrixColumn(matrix, 0);
        var up = new THREE.Vector3();
        up.setFromMatrixColumn(matrix, 1);
        var scale = targetDistance * Math.tan(scope.camera.fov * 0.5 * Math.PI / 180.0) / element.clientHeight;
        right.multiplyScalar(-deltaX * scale);
        up.multiplyScalar(deltaY * scale);
        tempTarget.add(right);
        tempTarget.add(up);
        panOffset.subVectors(tempTarget, target);
      }} else {{
        var offset = new THREE.Vector3();
        var te = scope.camera.matrix.elements;
        offset.x = -deltaX * te[0] - deltaY * te[4];
        offset.y = -deltaX * te[1] - deltaY * te[5];
        offset.z = -deltaX * te[2] - deltaY * te[6];
        offset.normalize();
        var distance = Math.sqrt(scope.camera.position.x * scope.camera.position.x + scope.camera.position.y * scope.camera.position.y + scope.camera.position.z * scope.camera.position.z);
        offset.multiplyScalar(distance * 0.001);
        panOffset.add(offset);
      }}
    }}
    function dollyIn(dollyScale) {{
      scale /= dollyScale;
    }}
    function dollyOut(dollyScale) {{
      scale *= dollyScale;
    }}
    this.getPolarAngle = getPolarAngle;
    this.getAzimuthalAngle = getAzimuthalAngle;
    this.rotateLeft = rotateLeft;
    this.rotateUp = rotateUp;
    this.panLeft = panLeft;
    this.panUp = panUp;
    this.pan = pan;
    this.dollyIn = dollyIn;
    this.dollyOut = dollyOut;
    this.update = function () {{
      var offset = new THREE.Vector3();
      offset.copy(scope.camera.position).sub(scope.target);
      spherical.setFromVector3(offset);
      spherical.theta += sphericalDelta.theta;
      spherical.phi += sphericalDelta.phi;
      spherical.theta = Math.max(scope.minAzimuthAngle, Math.min(scope.maxAzimuthAngle, spherical.theta));
      spherical.phi = Math.max(scope.minPolarAngle, Math.min(scope.maxPolarAngle, spherical.phi));
      spherical.makeSafe();
      spherical.radius *= scale;
      spherical.radius = Math.max(scope.minDistance, Math.min(scope.maxDistance, spherical.radius));
      offset.setFromSpherical(spherical);
      scope.camera.position.copy(scope.target).add(offset);
      scope.camera.lookAt(scope.target);
      if (scope.enableDamping === true) {{
        sphericalDelta.theta *= (1 - scope.dampingFactor);
        sphericalDelta.phi *= (1 - scope.dampingFactor);
        scale = 1 + (scale - 1) * (1 - scope.dampingFactor);
      }} else {{
        sphericalDelta.set(0, 0, 0);
        scale = 1;
      }}
      scope.camera.position.add(panOffset);
      scope.target.add(panOffset);
      panOffset.set(0, 0, 0);
      if (scope.autoRotate === true && currentState === state.NONE) {{
        rotateLeft(scope.autoRotateSpeed / 60 * Math.PI / 180);
      }}
      if (lastPosition.distanceToSquared(scope.camera.position) > EPS) {{
        lastPosition.copy(scope.camera.position);
        return true;
      }}
      return false;
    }};
    this.dispose = function () {{
      scope.domElement.removeEventListener('contextmenu', onContextMenu, false);
      scope.domElement.removeEventListener('mousedown', onMouseDown, false);
      scope.domElement.removeEventListener('wheel', onMouseWheel, false);
      scope.domElement.removeEventListener('touchstart', onTouchStart, false);
      scope.domElement.removeEventListener('touchend', onTouchEnd, false);
      scope.domElement.removeEventListener('touchmove', onTouchMove, false);
      window.removeEventListener('mousemove', onMouseMove, false);
      window.removeEventListener('mouseup', onMouseUp, false);
      window.removeEventListener('keydown', onKeyDown, false);
    }};
    function onContextMenu(event) {{
      if (scope.enabled === true) {{
        event.preventDefault();
      }}
    }}
    function onMouseDown(event) {{
      if (scope.enabled === false) return;
      if (event.button === 0 && scope.mouseButtons.LEFT === THREE.MOUSE.ROTATE ||
          event.button === 1 && scope.mouseButtons.MIDDLE === THREE.MOUSE.DOLLY ||
          event.button === 2 && scope.mouseButtons.RIGHT === THREE.MOUSE.PAN) {{
        event.preventDefault();
        mouseDownPosition.set(event.clientX, event.clientY);
        currentState = (event.button === 0) ? state.ROTATE : (event.button === 1) ? state.DOLLY : state.PAN;
        scope.domElement.addEventListener('mousemove', onMouseMove, false);
        window.addEventListener('mouseup', onMouseUp, false);
      }}
    }}
    function onMouseMove(event) {{
      if (scope.enabled === false) return;
      event.preventDefault();
      var movementX = event.movementX || event.mozMovementX || event.webkitMovementX || 0;
      var movementY = event.movementY || event.mozMovementY || event.webkitMovementY || 0;
      if (currentState === state.ROTATE && scope.enableRotate === true) {{
        var element = scope.domElement === document ? scope.domElement.body : scope.domElement;
        rotateLeft(2 * Math.PI * movementX / element.clientHeight * scope.rotateSpeed);
        rotateUp(2 * Math.PI * movementY / element.clientHeight * scope.rotateSpeed);
      }} else if (currentState === state.PAN && scope.enablePan === true) {{
        pan(movementX, movementY);
      }}
    }}
    function onMouseUp(event) {{
      if (scope.enabled === false) return;
      currentState = state.NONE;
      scope.domElement.removeEventListener('mousemove', onMouseMove, false);
      window.removeEventListener('mouseup', onMouseUp, false);
    }}
    function onMouseWheel(event) {{
      if (scope.enabled === false || scope.enableZoom === false) return;
      event.preventDefault();
      var delta = 0;
      if (event.deltaY) {{
        delta = event.deltaY;
      }} else if (event.wheelDelta) {{
        delta = -event.wheelDelta;
      }}
      if (delta > 0) {{
        dollyIn(1.1);
      }} else {{
        dollyOut(1.1);
      }}
    }}
    function onKeyDown(event) {{
      if (scope.enabled === false || scope.enablePan === false) return;
      switch (event.keyCode) {{
        case scope.keys.LEFT:
          panLeft(scope.keyPanSpeed);
          break;
        case scope.keys.RIGHT:
          panLeft(-scope.keyPanSpeed);
          break;
        case scope.keys.UP:
          panUp(scope.keyPanSpeed);
          break;
        case scope.keys.BOTTOM:
          panUp(-scope.keyPanSpeed);
          break;
      }}
    }}
    function onTouchStart(event) {{
      if (scope.enabled === false) return;
      switch (event.touches.length) {{
        case 1:
          touchStartTime = Date.now();
          touchStartPosition.set(event.touches[0].clientX, event.touches[0].clientY);
          currentState = state.TOUCH_ROTATE;
          break;
        case 2:
          var dx = event.touches[0].clientX - event.touches[1].clientX;
          var dy = event.touches[0].clientY - event.touches[1].clientY;
          touchStartDistance = Math.sqrt(dx * dx + dy * dy);
          currentState = state.TOUCH_DOLLY_PAN;
          break;
        default:
          currentState = state.NONE;
      }}
    }}
    function onTouchEnd(event) {{
      currentState = state.NONE;
    }}
    function onTouchMove(event) {{
      if (scope.enabled === false) return;
      event.preventDefault();
      switch (event.touches.length) {{
        case 1:
          if (currentState === state.TOUCH_ROTATE && scope.enableRotate === true) {{
            var deltaX = event.touches[0].clientX - touchStartPosition.x;
            var deltaY = event.touches[0].clientY - touchStartPosition.y;
            touchStartPosition.set(event.touches[0].clientX, event.touches[0].clientY);
            var element = scope.domElement === document ? scope.domElement.body : scope.domElement;
            rotateLeft(2 * Math.PI * deltaX / element.clientHeight * scope.rotateSpeed);
            rotateUp(2 * Math.PI * deltaY / element.clientHeight * scope.rotateSpeed);
          }}
          break;
        case 2:
          if (currentState === state.TOUCH_DOLLY_PAN) {{
            var dx = event.touches[0].clientX - event.touches[1].clientX;
            var dy = event.touches[0].clientY - event.touches[1].clientY;
            var distance = Math.sqrt(dx * dx + dy * dy);
            if (scope.enableZoom === true) {{
              var factor = touchStartDistance / distance;
              if (factor !== Infinity) {{
                scale *= factor;
              }}
            }}
            touchStartDistance = distance;
            if (scope.enablePan === true) {{
              var x = (event.touches[0].clientX + event.touches[1].clientX) / 2;
              var y = (event.touches[0].clientY + event.touches[1].clientY) / 2;
              var deltaX = x - touchStartPosition.x;
              var deltaY = y - touchStartPosition.y;
              touchStartPosition.set(x, y);
              pan(deltaX, deltaY);
            }}
          }}
          break;
      }}
    }}
    scope.domElement.addEventListener('contextmenu', onContextMenu, false);
    scope.domElement.addEventListener('mousedown', onMouseDown, false);
    scope.domElement.addEventListener('wheel', onMouseWheel, false);
    scope.domElement.addEventListener('touchstart', onTouchStart, false);
    scope.domElement.addEventListener('touchend', onTouchEnd, false);
    scope.domElement.addEventListener('touchmove', onTouchMove, false);
    window.addEventListener('keydown', onKeyDown, false);
    spherical.setFromVector3(scope.camera.position.sub(scope.target));
    lastPosition.copy(scope.camera.position);
  }};
  // End of OrbitControls implementation
  </script>
</head>
<body>
<script>
const initialState = {json.dumps(initial_state)};
const moves = {json.dumps(moves)};
const orders = {json.dumps(indince)};

let scene, camera, renderer, controls;
let stickers = {{}}; // 保存每个小贴片的Mesh

init();
animate();

function init() {{
    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(6, 6, 6);

    renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    // 轨道控制器
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 3;
    controls.maxDistance = 20;
    controls.update();

    // 灯光
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(5, 10, 7);
    scene.add(dirLight);

    // 创建魔方
    createCube(initialState, orders);

    // 窗口自适应
    window.addEventListener('resize', onWindowResize, false);

    // 播放颜色变化
    playMoves(moves, 1600);
}}
function makeNumberTexture(number, size = 128) {{
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');

    // 背景透明
    ctx.clearRect(0, 0, size, size);

    // 字体样式
    ctx.fillStyle = 'black';
    ctx.font = `${{size * 0.7}}px Arial`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // 画数字
    ctx.fillText(number.toString(), size / 2, size / 2);

    // 创建纹理
    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    return texture;
}}

function createCube(state, orders) {{
    const stickerSize = 0.9;
    const gap = 0.05;
    const offset = (stickerSize + gap);

    // 每个面的方向和偏移
    const faceConfig = {{
        U: {{ normal: [0, 1, 0], base: [0, 1.5, 0], udir: [1, 0, 0], vdir: [0, 0, -1] }},
        D: {{ normal: [0, -1, 0], base: [0, -1.5, 0], udir: [1, 0, 0], vdir: [0, 0, 1] }},
        F: {{ normal: [0, 0, 1], base: [0, 0, 1.5], udir: [1, 0, 0], vdir: [0, -1, 0] }},
        B: {{ normal: [0, 0, -1], base: [0, 0, -1.5], udir: [-1, 0, 0], vdir: [0, -1, 0] }},
        L: {{ normal: [-1, 0, 0], base: [-1.5, 0, 0], udir: [0, 0, -1], vdir: [0, -1, 0] }},
        R: {{ normal: [1, 0, 0], base: [1.5, 0, 0], udir: [0, 0, 1], vdir: [0, -1, 0] }},
    }};

    for (let face in state) {{
      let colors = state[face];
      let cfg = faceConfig[face];
      let numbers = orders[face];
      for (let i = 0; i < 9; i++) {{
        let row = Math.floor(i / 3);
        let col = i % 3;
        let centerOffsetU = (col - 1) * offset;
        let centerOffsetV = (row - 1) * offset;

        let px = cfg.base[0] + cfg.udir[0] * centerOffsetU + cfg.vdir[0] * centerOffsetV;
        let py = cfg.base[1] + cfg.udir[1] * centerOffsetU + cfg.vdir[1] * centerOffsetV;
        let pz = cfg.base[2] + cfg.udir[2] * centerOffsetU + cfg.vdir[2] * centerOffsetV;

        // 颜色贴图
        let colorMaterial = new THREE.MeshLambertMaterial({{ color: colors[i] }});

        // 数字贴图
        let numberTexture = makeNumberTexture(numbers[i]);
        let numberMaterial = new THREE.MeshBasicMaterial({{ map: numberTexture, transparent: true }});

        // 创建面片（贴纸）
        let geometry = new THREE.PlaneGeometry(stickerSize, stickerSize);

        // 创建颜色面
        let sticker = new THREE.Mesh(geometry, colorMaterial);

        // 创建数字面，稍微抬高避免 z-fighting
        let numberPlane = new THREE.Mesh(geometry, numberMaterial);
        numberPlane.position.x += cfg.normal[0] * 0.1;
        numberPlane.position.y += cfg.normal[1] * 0.1;
        numberPlane.position.z += cfg.normal[2] * 0.1;

        // 旋转对齐
        let normal = new THREE.Vector3(...cfg.normal);
        let up = new THREE.Vector3(...cfg.vdir).negate();
        let quaternion = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal);
        sticker.quaternion.multiply(quaternion);
        numberPlane.quaternion.multiply(quaternion);

        // 设置位置
        sticker.position.set(px, py, pz);
        numberPlane.position.set(px, py, pz + 0.01);

        scene.add(sticker);
        scene.add(numberPlane);

        stickers[face + i] = sticker;
      }}
    }}
}}

function updateColors(state) {{
    for (let key in stickers) {{
        let face = key[0];
        let idx = parseInt(key.slice(1));
        stickers[key].material.color.set(state[face][idx]);
    }}
}}

function playMoves(moves, interval) {{
    let step = 0;
    setInterval(() => {{
        if (step < moves.length) {{
            updateColors(moves[step]);
            step++;
        }}
    }}, interval);
}}

function onWindowResize() {{
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}}

function animate() {{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}}
</script>
</body>
</html>
"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"已生成 {output_file} ，用浏览器打开即可。")


# ===== 测试示例 =====
if __name__ == "__main__":
    initial_state = {
        "U": ["white"] * 9,
        "D": ["yellow"] * 9,
        "F": ["red"] * 9,
        "B": ["orange"] * 9,
        "L": ["blue"] * 9,
        "R": ["green"] * 9
    }

    # 生成几个动作后的状态（这里只是演示，实际可根据魔方动作计算）
    move1 = initial_state.copy()
    move1 = {f: c[:] for f, c in move1.items()}
    move1["F"][0] = "blue"

    move2 = {f: c[:] for f, c in move1.items()}
    move2["R"][4] = "white"

    moves = [move1, move2]

    generate_rubiks_html(initial_state, moves)
