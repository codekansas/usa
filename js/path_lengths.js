/**
 * Controls the visualization of the path lengths for different planners.
 */

$(document).ready(function () {
  let scene;
  let camera;
  let renderer;
  let center;
  let loader;
  let sceneMesh;
  let scenePathMesh;

  let currentScene;
  let currentPath;

  PATH_LENGTH_PLANNERS = [
    {
      key: "gradient_clip_sdf",
      name: "Gradient",
    },
    {
      key: "a_star_10_cm_clip_sdf",
      name: "Grid",
    },
    // {
    //   key: "a_star_20_cm_clip_sdf",
    //   name: "Grid (20 cm)",
    // },
    // {
    //   key: "a_star_30_cm_clip_sdf",
    //   name: "Grid (30 cm)",
    // },
    // {
    //   key: "a_star_40_cm_clip_sdf",
    //   name: "Grid (40 cm)",
    // },
    {
      key: "a_star_10_cm_occ_map",
      name: "Occupancy Map",
    },
  ];

  PATH_LENGTH_PATHS = {
    kitchen_stretch: {
      name: "Kitchen",
      key: "kitchen_stretch",
      paths: [
        "0.png",
        "1.png",
        "2.png",
        "3.png",
        "4.png",
        "5.png",
        "6.png",
        "7.png",
        "8.png",
        "9.png",
      ],
    },
    chess_stretch: {
      name: "Chess",
      key: "chess_stretch",
      paths: [
        "0.png",
        "1.png",
        "2.png",
        "3.png",
        "4.png",
        "5.png",
        "6.png",
        "7.png",
        "8.png",
        "9.png",
      ],
    },
    lab_stretch: {
      name: "Lab",
      key: "lab_stretch",
      paths: [
        "0.png",
        "1.png",
        "2.png",
        "3.png",
        "4.png",
        "5.png",
        "6.png",
        "7.png",
        "8.png",
        "9.png",
      ],
    },
    lab_r3d: {
      name: "Lab (Record3D)",
      key: "lab_r3d",
      paths: [
        "0.png",
        "1.png",
        "2.png",
        "3.png",
        "4.png",
        "5.png",
        "6.png",
        "7.png",
        "8.png",
        "9.png",
      ],
    },
    studio_r3d: {
      name: "Studio (Record3D)",
      key: "studio_r3d",
      paths: [
        "0.png",
        "1.png",
        "2.png",
        "3.png",
        "4.png",
        "5.png",
        "6.png",
        "7.png",
        "8.png",
        "9.png",
      ],
    },
    replica_apt_3_mnp: {
      name: "Replica",
      key: "replica_apt_3_mnp",
      paths: [
        "0.png",
        "1.png",
        "2.png",
        "3.png",
        "4.png",
        "5.png",
        "6.png",
        "7.png",
        "8.png",
        "9.png",
      ],
    },
  };

  function setPath(path) {
    currentPath = path;

    // Adds images for all planners.
    const pathImagesDiv = document.getElementById("path-lengths-images");
    pathImagesDiv.innerHTML = "";

    for (const planner of PATH_LENGTH_PLANNERS) {
      const column = document.createElement("div");
      column.classList.add("column");
      pathImagesDiv.appendChild(column);

      const addImage = (srcRef) => {
        const figure = document.createElement("figure");
        figure.classList.add("image");

        const imgLink = document.createElement("a");
        imgLink.href = srcRef;
        imgLink.target = "_blank";

        const img = document.createElement("img");
        img.src = srcRef;
        img.width = 256;
        img.height = 256;
        img.classList.add("has-ratio");

        const figcaption = document.createElement("figcaption");
        figcaption.innerText = planner.name;

        imgLink.appendChild(img);
        figure.appendChild(imgLink);
        figure.appendChild(figcaption);
        column.appendChild(figure);
      };

      addImage(
        `assets/${currentScene.key}/path_lengths/${planner.key}/${path}`
      );

      addImage(`assets/${currentScene.key}/path_lengths/${planner.key}.png`);
    }
  }

  function setScene(scn) {
    currentScene = scn;
    setPath(currentScene.paths[0]);
    const plyUrl = `assets/${scn.key}/point_cloud.ply`;
    const pathPlyUrl = `assets/${scn.key}/path_point_cloud.ply`;

    loader.load(plyUrl, function (geometry) {
      const material = new THREE.PointsMaterial({
        size: 0.1,
        vertexColors: THREE.VertexColors,
      });
      geometry.computeBoundingBox();
      center = new THREE.Vector3();
      geometry.boundingBox.getCenter(center);
      if (sceneMesh) {
        scene.remove(sceneMesh);
      }
      sceneMesh = new THREE.Points(geometry, material);
      scene.add(sceneMesh);

      // Translates the scene to the center of the bounding box.
      scene.position.x = -center.x;
      scene.position.y = -center.y;
      scene.position.z = -center.z;

      // For the given scene, add buttons for each path.
      const pathButtonDiv = document.getElementById(
        "path-lengths-point-cloud-path-buttons"
      );
      pathButtonDiv.innerHTML = "";
      for (let i = 0; i < scn.paths.length; i++) {
        const path = scn.paths[i];
        const button = document.createElement("button");
        button.classList.add("button");
        button.style.margin = "0.5rem";
        button.innerHTML = `${i}`;
        button.onclick = function () {
          setPath(path);
        };
        pathButtonDiv.appendChild(button);
      }
    });

    loader.load(pathPlyUrl, function (geometry) {
      const material = new THREE.PointsMaterial({
        size: 0.1,
        vertexColors: THREE.VertexColors,
      });
      if (scenePathMesh) {
        scene.remove(scenePathMesh);
      }
      scenePathMesh = new THREE.Points(geometry, material);
      scene.add(scenePathMesh);
    });
  }

  function init() {
    const container = document.getElementById("path-lengths-point-cloud");

    scene = new THREE.Scene();

    // const aspectRatio = container.clientHeight / container.clientWidth;
    const aspectRatio = 2 / 1;
    camera = new THREE.PerspectiveCamera(75, aspectRatio, 0.1, 1000);
    camera.position.z = 10;

    renderer = new THREE.WebGLRenderer({ alpha: true });
    renderer.setClearColor(0xffffff, 0);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.zoomSpeed = 0.3;

    // Instantiates loader.
    loader = new THREE.PLYLoader();

    // Sets the initial scene.
    setScene(PATH_LENGTH_PATHS.kitchen_stretch);
    setPath(PATH_LENGTH_PATHS.kitchen_stretch.paths[0]);

    // Adds buttons for each of the scenes.
    const sceneButtonsDiv = document.getElementById(
      "path-lengths-point-cloud-scene-buttons"
    );
    for (const [key, value] of Object.entries(PATH_LENGTH_PATHS)) {
      const button = document.createElement("button");
      button.innerText = value.name;
      button.classList.add("button");
      button.style.margin = "0.5rem";
      button.addEventListener("click", () => {
        setScene(value);
      });
      sceneButtonsDiv.appendChild(button);
    }

    // Adds the renderer to the container.
    container.appendChild(renderer.domElement);

    animate();
  }

  function animate() {
    const container = document.getElementById("path-lengths-point-cloud");
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }

  init();
});
