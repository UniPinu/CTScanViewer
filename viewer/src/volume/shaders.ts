/**
 * GPU ray-marching through a 3-D texture of the CT volume.
 *
 * The mesh is a unit cube (-0.5..0.5) whose model matrix scales it to the
 * physical extent of the scan. Rays are traced in that unit-cube space, and
 * converted to texture coordinates such that:
 *   texture x = column  -> world X (patient's left = +X)
 *   texture y = row     -> world -Z (anterior/front of chest = +Z)
 *   texture z = slice   -> world Y (head = +Y)
 *
 * Voxels are stored as uint8 with HU = v*2550 - 1000 (10 HU per level),
 * see VolumeView.buildTexture.
 */

export const vertexShader = /* glsl */ `
out vec3 vOrigin;
out vec3 vDirection;

void main() {
  vOrigin = (inverse(modelMatrix) * vec4(cameraPosition, 1.0)).xyz;
  vDirection = position - vOrigin;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

export const fragmentShader = /* glsl */ `
precision highp float;
precision highp sampler3D;

uniform sampler3D uVolume;
uniform int   uMode;        // 0 = MIP, 1 = isosurface, 2 = composite layers
uniform float uSteps;       // samples along the cube diagonal
uniform float uIso;         // isosurface threshold, normalised 0..1
uniform vec2  uWindow;      // MIP window lo/hi, normalised 0..1
uniform vec3  uOpacity;     // composite opacity multipliers: lung, soft tissue, bone
uniform bool  uClipOn;      // hide everything above the cutting plane
uniform float uClipY;       // cutting plane height in unit-cube space

in vec3 vOrigin;
in vec3 vDirection;
out vec4 fragColor;

vec2 hitBox(vec3 orig, vec3 dir) {
  vec3 invDir = 1.0 / dir;
  vec3 t0s = (vec3(-0.5) - orig) * invDir;
  vec3 t1s = (vec3( 0.5) - orig) * invDir;
  vec3 tmin = min(t0s, t1s);
  vec3 tmax = max(t0s, t1s);
  return vec2(max(tmin.x, max(tmin.y, tmin.z)), min(tmax.x, min(tmax.y, tmax.z)));
}

vec3 toTex(vec3 p) { return vec3(p.x + 0.5, 0.5 - p.z, p.y + 0.5); }
float sampleAt(vec3 p) { return texture(uVolume, toTex(p)).r; }
float toHU(float v) { return v * 2550.0 - 1000.0; }

vec3 gradientAt(vec3 p, float h) {
  return vec3(
    sampleAt(p + vec3(h, 0, 0)) - sampleAt(p - vec3(h, 0, 0)),
    sampleAt(p + vec3(0, h, 0)) - sampleAt(p - vec3(0, h, 0)),
    sampleAt(p + vec3(0, 0, h)) - sampleAt(p - vec3(0, 0, h)));
}

// Simple headlight + rim shading for surfaces.
vec3 shade(vec3 base, vec3 p, vec3 rayDir) {
  vec3 n = -gradientAt(p, 1.0 / 256.0);
  if (length(n) < 1e-5) return base * 0.6;
  n = normalize(n);
  if (dot(n, -rayDir) < 0.0) n = -n;
  float diff = max(dot(n, -rayDir), 0.0);
  vec3 side = normalize(vec3(0.6, 0.8, 0.4));
  float fill = max(dot(n, side), 0.0);
  return base * (0.22 + 0.6 * diff + 0.25 * fill);
}

// Colour + opacity per HU band. Opacities are tuned for 400 steps and
// rescaled to the actual step count below.
vec4 transfer(float hu) {
  if (hu < -900.0) return vec4(0.0);
  if (hu < -500.0) return vec4(0.30, 0.55, 1.00, 0.0025 * uOpacity.x);
  if (hu <  150.0) return vec4(0.95, 0.62, 0.48, 0.0060 * uOpacity.y);
  return vec4(1.00, 0.97, 0.88, 0.25 * uOpacity.z);
}

void main() {
  vec3 rayDir = normalize(vDirection);
  vec2 b = hitBox(vOrigin, rayDir);
  if (b.x > b.y) discard;
  b.x = max(b.x, 0.0);

  // Restrict the ray to the half-space below the cutting plane.
  if (uClipOn) {
    if (abs(rayDir.y) < 1e-6) {
      if (vOrigin.y > uClipY) discard;
    } else {
      float tPlane = (uClipY - vOrigin.y) / rayDir.y;
      if (rayDir.y > 0.0) b.y = min(b.y, tPlane); else b.x = max(b.x, tPlane);
    }
    if (b.x >= b.y) discard;
  }

  float dt = 1.7320508 / uSteps;
  float alphaScale = 400.0 / uSteps;
  vec3 p = vOrigin + b.x * rayDir;

  if (uMode == 0) {
    float m = 0.0;
    for (float t = b.x; t < b.y; t += dt) {
      m = max(m, sampleAt(p));
      p += rayDir * dt;
    }
    float g = clamp((m - uWindow.x) / max(uWindow.y - uWindow.x, 1e-4), 0.0, 1.0);
    fragColor = vec4(vec3(g), g);
    return;
  }

  if (uMode == 1) {
    for (float t = b.x; t < b.y; t += dt) {
      if (sampleAt(p) >= uIso) {
        // Refine the hit a little for smoother surfaces.
        vec3 q = p - rayDir * dt * 0.5;
        if (sampleAt(q) >= uIso) p = q;
        vec3 base = toHU(uIso) > 150.0 ? vec3(0.96, 0.94, 0.86) : vec3(0.93, 0.68, 0.56);
        fragColor = vec4(shade(base, p, rayDir), 1.0);
        return;
      }
      p += rayDir * dt;
    }
    discard;
  }

  vec4 acc = vec4(0.0);
  for (float t = b.x; t < b.y; t += dt) {
    vec4 c = transfer(toHU(sampleAt(p)));
    if (c.a > 0.0) {
      c.a = 1.0 - pow(1.0 - c.a, alphaScale);
      if (c.a > 0.05) c.rgb = shade(c.rgb, p, rayDir) * 1.4;
      acc.rgb += (1.0 - acc.a) * c.a * c.rgb;
      acc.a   += (1.0 - acc.a) * c.a;
      if (acc.a > 0.98) break;
    }
    p += rayDir * dt;
  }
  fragColor = acc;
}
`;
