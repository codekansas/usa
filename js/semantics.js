/**
 * Controls the visualization for the semantic path planning.
 */

$(document).ready(function () {
  let scene;
  let camera;
  let renderer;
  let center;
  let loader;
  let sceneMesh;
  let scenePathMesh;
  let pathMesh;
  let topPointsMesh;

  let currentScene;
  let currentPath;
  let currentPlanner;

  SEMANTIC_PLANNERS = [
    {
      key: "gradient_clip_sdf",
      name: "Gradient",
    },
    {
      key: "a_star_10_cm_clip_sdf",
      name: "Grid (10 cm)",
    },
    {
      key: "a_star_20_cm_clip_sdf",
      name: "Grid (20 cm)",
    },
    {
      key: "a_star_30_cm_clip_sdf",
      name: "Grid (30 cm)",
    },
    {
      key: "a_star_40_cm_clip_sdf",
      name: "Grid (40 cm)",
    },
  ];

  SEMANTIC_PATHS = {
    kitchen_stretch: {
      name: "Kitchen",
      key: "kitchen_stretch",
      paths: [
        {
          name: "A refrigerator full of beverages",
          path: "a_refrigerator_full_of_beverages",
        },
        {
          name: "Some snacks",
          path: "some_snacks",
        },
        {
          name: "The coffee machine",
          path: "the_coffee_machine",
        },
        {
          name: "The row of stools",
          path: "the_row_of_stools",
        },
        {
          name: "The man sitting at a laptop",
          path: "the_man_sitting_at_a_laptop",
        },
      ],
    },
    chess_stretch: {
      name: "Chess",
      key: "chess_stretch",
      paths: [
        {
          name: "A chess board",
          path: "a_chess_board",
        },
        {
          name: "A comfortable chair",
          path: "a_comfortable_chair",
        },
        {
          name: "A conference room",
          path: "a_conference_room",
        },
      ],
    },
    lab_stretch: {
      name: "Lab",
      key: "lab_stretch",
      paths: [
        {
          name: "A man sitting at a computer",
          path: "a_man_sitting_at_a_computer",
        },
        {
          name: "Computer desk chair",
          path: "computer_desk_chair",
        },
        {
          name: "A wooden box",
          path: "a_wooden_box",
        },
        {
          name: "Desktop computer",
          path: "desktop_computer",
        },
        {
          name: "Doorway",
          path: "doorway",
        },
        {
          name: "Shelves",
          path: "shelves",
        },
      ],
    },
    lab_r3d: {
      name: "Lab (Record3D)",
      key: "lab_r3d",
      paths: [
        // {
        //   name: "A computer desk",
        //   path: "a_computer_desk",
        // },
        // {
        //   name: "Computer desk chair",
        //   path: "computer_desk_chair",
        // },
        {
          name: "A wooden box",
          path: "a_wooden_box",
        },
        // {
        //   name: "Desktop computer",
        //   path: "desktop_computer",
        // },
        {
          name: "Doorway",
          path: "doorway",
        },
        {
          name: "Shelves",
          path: "shelves",
        },
      ],
    },
    studio_r3d: {
      name: "Studio (Record3D)",
      key: "studio_r3d",
      paths: [
        {
          name: "The drill press",
          path: "the_drill_press",
        },
        {
          name: "A cabinet with green organization boxes",
          path: "a_cabinet_with_green_organization_boxes",
        },
        {
          name: "The mitre saw",
          path: "the_mitre_saw",
        },
      ],
    },
    replica_apt_3_mnp: {
      name: "Replica",
      key: "replica_apt_3_mnp",
      paths: [
        {
          name: "Jackets hanging in the closet",
          path: "jackets_hanging_in_the_closet",
        },
        {
          name: "The bicycle",
          path: "the_bicycle",
        },
        {
          name: "A kitchen sink",
          path: "a_kitchen_sink",
        },
        {
          name: "A video game kitchen sink",
          path: "a_video_game_kitchen_sink",
        },
      ],
    },
  };

  function setScene(scn) {
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

    // Sets the video to the scene.
    const videoContainerDiv = document.getElementById("semantics-video");
    videoContainerDiv.innerHTML = "";
    const video = document.createElement("video");
    video.setAttribute("id", "video");
    video.setAttribute("width", "100%");
    video.setAttribute("autoplay", "true");
    video.setAttribute("muted", "true");
    video.setAttribute("playsinline", "true");
    video.setAttribute("loop", "true");
    video.setAttribute("controls", "true");
    video.setAttribute("src", `assets/${scn.key}/video.mp4`);
    videoContainerDiv.appendChild(video);
  }

  function setPath(scn, path, planner) {
    const plyUrl = `assets/${scn.key}/semantics/${planner.key}/${path.path}_path.ply`;
    const topPointsPlyUrl = `assets/${scn.key}/semantics/${planner.key}/${path.path}_top.ply`;

    // Loads the path.
    loader.load(plyUrl, function (geometry) {
      const material = new THREE.PointsMaterial({
        size: 0.1,
        vertexColors: THREE.VertexColors,
      });
      if (pathMesh) {
        scene.remove(pathMesh);
      }
      pathMesh = new THREE.Points(geometry, material);
      scene.add(pathMesh);
      document.getElementById("semantics-point-cloud-label").innerText =
        path.name;
    });

    // Loads the top N points.
    loader.load(topPointsPlyUrl, function (geometry) {
      const material = new THREE.PointsMaterial({
        size: 0.3,
        vertexColors: THREE.VertexColors,
      });
      if (topPointsMesh) {
        scene.remove(topPointsMesh);
      }
      topPointsMesh = new THREE.Points(geometry, material);
      scene.add(topPointsMesh);
    });
  }

  function setScenePathButtons(scene) {
    const pathButtonsDiv = document.getElementById(
      "semantics-point-cloud-path-buttons"
    );
    pathButtonsDiv.innerHTML = "";
    for (const path of Object.values(scene.paths)) {
      const button = document.createElement("button");
      button.innerText = path.name;
      button.classList.add("button");
      button.style.margin = "0.5rem";
      button.addEventListener("click", () => {
        currentPath = path;
        refreshSemanticsPath();
      });
      pathButtonsDiv.appendChild(button);
    }
  }

  function refreshSemanticsPath() {
    setPath(currentScene, currentPath, currentPlanner);
  }

  function refreshSemanticsScene() {
    setScene(currentScene);
    setPath(currentScene, currentPath, currentPlanner);
    setScenePathButtons(currentScene);
  }

  function initSemantics() {
    const container = document.getElementById("semantics-point-cloud");

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
    currentPath = SEMANTIC_PATHS.kitchen_stretch.paths[0];
    currentScene = SEMANTIC_PATHS.kitchen_stretch;
    currentPlanner = SEMANTIC_PLANNERS[0];
    document.getElementById("semantics-planner-label").innerText =
      currentPlanner.name;
    refreshSemanticsScene();

    // Adds buttons for each of the planners.
    const plannerButtonsDiv = document.getElementById(
      "semantics-point-cloud-planner-buttons"
    );
    for (const planner of SEMANTIC_PLANNERS) {
      const button = document.createElement("button");
      button.innerText = planner.name;
      button.classList.add("button");
      button.style.margin = "0.5rem";
      button.addEventListener("click", () => {
        currentPlanner = planner;
        document.getElementById("semantics-planner-label").innerText =
          currentPlanner.name;
        refreshSemanticsPath();
      });
      plannerButtonsDiv.appendChild(button);
    }

    // Adds buttons for each of the scenes.
    const sceneButtonsDiv = document.getElementById(
      "semantics-point-cloud-scene-buttons"
    );
    for (const [key, value] of Object.entries(SEMANTIC_PATHS)) {
      const button = document.createElement("button");
      button.innerText = value.name;
      button.classList.add("button");
      button.style.margin = "0.5rem";
      button.addEventListener("click", () => {
        currentScene = value;
        currentPath = value.paths[0];
        refreshSemanticsScene();
      });
      sceneButtonsDiv.appendChild(button);
    }

    // Adds the renderer to the container.
    container.appendChild(renderer.domElement);

    animateSemantics();
  }

  function animateSemantics() {
    const container = document.getElementById("semantics-point-cloud");
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.render(scene, camera);
    requestAnimationFrame(animateSemantics);
  }

  initSemantics();
});
