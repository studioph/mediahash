{ pkgs ? import <nixpkgs> {} }:

(pkgs.buildFHSEnv {
  name = "mediahash";
  targetPkgs = pkgs: (with pkgs; [
    gcc
    gdb
    ffmpeg-headless
    zlib
    pkg-config
    python311Packages.cython
    python311Packages.setuptools
  ]);
  profile = ''
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [pkgs.zlib]}:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib.outPath}/lib:$LD_LIBRARY_PATH"
  '';
}).env