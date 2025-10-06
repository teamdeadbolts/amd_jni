{ pkgs ? import <nixpkgs> {} }:
let
  # Pin to a nixpkgs version that has OpenCV 4.8.0
  # or override the version directly
  opencv4100 = pkgs.opencv4.overrideAttrs (oldAttr: rec {
    version = "4.10.0";
    src = pkgs.fetchFromGitHub {
      owner = "opencv";
      repo = "opencv";
      rev = version;
      hash = "sha256-s+KvBrV/BxrxEvPhHzWCVFQdUQwhUdRJyb0wcGDFpeo=" ; 
    };
    
    nativeBuildInputs =
      oldAttr.nativeBuildInputs
      ++ (with pkgs; [
        ant
        openjdk
        python3
        python3Packages.numpy
      ]);
      
    cmakeFlags =
      oldAttr.cmakeFlags
      ++ [
        "-DBUILD_JAVA=ON"
        "-DBUILD_opencv_dnn=OFF"
      ];
  });
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    openjdk21
    cmake
    opencv4100
    clang
  ];
  
  shellHook = ''
    export LD_LIBRARY_PATH=${opencv4100}/share/java/opencv4:$LD_LIBRARY_PATH
  '';
}