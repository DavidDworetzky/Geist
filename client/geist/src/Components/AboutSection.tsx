import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useBranding } from '../branding';

interface SystemInfo {
  version: string;
  platform?: {
    system: string;
    release: string;
    machine: string;
  };
  python?: {
    version: string;
  };
  inference?: {
    mode: 'local' | 'online';
    engine: string;
    model: string;
    provider?: string | null;
    acceleration?: string | null;
  };
}

interface Detail {
  label: string;
  value: string;
}

const engineNames: Record<string, string> = {
  llama_server: 'llama.cpp',
  mlx_llama: 'MLX',
  transformers: 'Transformers',
};

function displayEngine(engine: string): string {
  return engineNames[engine] ?? engine;
}

function displayValue(value?: string | null): string {
  return value?.trim() || 'Not reported';
}

function AboutAuroraBackground(): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || typeof WebGLRenderingContext === 'undefined') {
      return undefined;
    }

    const gl = canvas.getContext('webgl', {
      alpha: false,
      antialias: false,
      depth: false,
      powerPreference: 'low-power',
    });
    if (!gl) {
      return undefined;
    }

    const vertexSource = `
      attribute vec2 aPosition;
      void main() {
        gl_Position = vec4(aPosition, 0.0, 1.0);
      }
    `;
    const fragmentSource = `
      precision highp float;
      uniform vec2 uResolution;
      uniform float uTime;
      uniform vec3 uDeepColor;
      uniform vec3 uSurfaceColor;
      uniform vec3 uAccentColor;

      float random(vec2 point) {
        return fract(sin(dot(point, vec2(127.1, 311.7))) * 43758.5453123);
      }

      float noise(vec2 point) {
        vec2 cell = floor(point);
        vec2 local = fract(point);
        vec2 blend = local * local * (3.0 - 2.0 * local);
        return mix(
          mix(random(cell), random(cell + vec2(1.0, 0.0)), blend.x),
          mix(random(cell + vec2(0.0, 1.0)), random(cell + vec2(1.0, 1.0)), blend.x),
          blend.y
        );
      }

      float flowingNoise(vec2 point) {
        float value = 0.0;
        float amplitude = 0.5;
        mat2 rotation = mat2(0.8, 0.6, -0.6, 0.8);
        for (int octave = 0; octave < 3; octave++) {
          value += noise(point) * amplitude;
          point = rotation * point * 2.02 + vec2(8.3, 3.7);
          amplitude *= 0.5;
        }
        return value;
      }

      float auroraRibbon(
        vec2 uv,
        float time,
        float center,
        float frequency,
        float amplitude,
        float width,
        float phase
      ) {
        float turbulence = flowingNoise(vec2(uv.x * 1.8 + phase, time * 0.09 + phase)) - 0.44;
        float wave = sin(uv.x * frequency + time * 0.58 + phase) * amplitude;
        wave += sin(uv.x * frequency * 0.46 - time * 0.34 + phase * 1.7) * amplitude * 0.52;
        wave += turbulence * amplitude * 1.35;

        float distanceToRibbon = abs(uv.y - center - wave);
        float core = exp(-distanceToRibbon * distanceToRibbon / (width * width));
        float bloom = exp(-distanceToRibbon * distanceToRibbon / (width * width * 6.0));
        return core * 0.72 + bloom * 0.28;
      }

      void main() {
        vec2 uv = gl_FragCoord.xy / uResolution.xy;
        float time = uTime * 0.46;

        float upperRibbon = auroraRibbon(uv, time, 0.7, 6.1, 0.12, 0.058, 0.4);
        float middleRibbon = auroraRibbon(uv, time, 0.48, 7.4, 0.15, 0.067, 2.35);
        float lowerRibbon = auroraRibbon(uv, time, 0.27, 5.3, 0.1, 0.052, 4.8);

        float atmosphere = flowingNoise(vec2(uv.x * 2.3 - time * 0.05, uv.y * 1.4 + time * 0.035));
        float verticalFade = smoothstep(0.0, 0.24, uv.y) * smoothstep(1.0, 0.7, uv.y);

        vec3 coolColor = mix(uAccentColor, vec3(0.34, 0.58, 1.0), 0.38);
        vec3 warmColor = mix(uAccentColor, vec3(1.0, 0.38, 0.82), 0.2);
        vec3 color = mix(uDeepColor, uSurfaceColor, 0.28 + uv.y * 0.11);
        color += uAccentColor * upperRibbon * 0.2;
        color += coolColor * middleRibbon * 0.25;
        color += warmColor * lowerRibbon * 0.17;
        color += uAccentColor * max(atmosphere - 0.5, 0.0) * 0.07 * verticalFade;

        float centerGlow = exp(-length((uv - vec2(0.32, 0.52)) * vec2(0.72, 1.0)) * 2.8);
        color += uAccentColor * centerGlow * 0.045;

        float vignette = smoothstep(0.92, 0.3, length((uv - 0.5) * vec2(0.72, 1.0)));
        color *= 0.82 + vignette * 0.18;
        gl_FragColor = vec4(color, 1.0);
      }
    `;

    const compileShader = (type: number, source: string) => {
      const shader = gl.createShader(type);
      if (!shader) {
        return null;
      }
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        gl.deleteShader(shader);
        return null;
      }
      return shader;
    };

    const vertexShader = compileShader(gl.VERTEX_SHADER, vertexSource);
    const fragmentShader = compileShader(gl.FRAGMENT_SHADER, fragmentSource);
    const program = gl.createProgram();
    if (!vertexShader || !fragmentShader || !program) {
      return undefined;
    }

    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      gl.deleteProgram(program);
      return undefined;
    }

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW,
    );
    gl.useProgram(program);
    const position = gl.getAttribLocation(program, 'aPosition');
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);

    const resolution = gl.getUniformLocation(program, 'uResolution');
    const time = gl.getUniformLocation(program, 'uTime');
    const deepColor = gl.getUniformLocation(program, 'uDeepColor');
    const surfaceColor = gl.getUniformLocation(program, 'uSurfaceColor');
    const accentColor = gl.getUniformLocation(program, 'uAccentColor');

    const readColor = (token: string, fallback: [number, number, number]) => {
      const value = getComputedStyle(document.documentElement).getPropertyValue(token).trim();
      const match = value.match(/^#([0-9a-f]{6})$/i);
      if (!match) {
        return fallback;
      }
      const numeric = Number.parseInt(match[1], 16);
      return [
        ((numeric >> 16) & 255) / 255,
        ((numeric >> 8) & 255) / 255,
        (numeric & 255) / 255,
      ] as [number, number, number];
    };

    const deep = readColor('--geist-color-bg', [0.06, 0.025, 0.12]);
    const surface = readColor('--geist-color-surface-strong', [0.22, 0.09, 0.43]);
    const accent = readColor('--geist-color-accent', [0.65, 0.55, 0.98]);
    gl.uniform3fv(deepColor, deep);
    gl.uniform3fv(surfaceColor, surface);
    gl.uniform3fv(accentColor, accent);

    const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
    let animationFrame = 0;
    const render = (timestamp: number) => {
      const bounds = canvas.getBoundingClientRect();
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.max(1, Math.round(bounds.width * pixelRatio));
      const height = Math.max(1, Math.round(bounds.height * pixelRatio));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
        gl.viewport(0, 0, width, height);
      }

      gl.uniform2f(resolution, width, height);
      gl.uniform1f(time, reduceMotion ? 0 : timestamp / 1000);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      if (!reduceMotion) {
        animationFrame = window.requestAnimationFrame(render);
      }
    };
    animationFrame = window.requestAnimationFrame(render);

    return () => {
      window.cancelAnimationFrame(animationFrame);
      gl.deleteBuffer(buffer);
      gl.deleteProgram(program);
      gl.deleteShader(vertexShader);
      gl.deleteShader(fragmentShader);
    };
  }, []);

  return <canvas ref={canvasRef} className="about-aurora-background" aria-hidden="true" />;
}

export default function AboutSection(): JSX.Element {
  const branding = useBranding();
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    fetch('/api/v1/system')
      .then((response) => {
        if (!response.ok) {
          throw new Error(`System information failed: ${response.statusText}`);
        }
        return response.json() as Promise<SystemInfo>;
      })
      .then((payload) => {
        if (!cancelled) {
          setSystemInfo(payload);
          setError(null);
        }
      })
      .catch((requestError) => {
        if (!cancelled) {
          setError(requestError instanceof Error ? requestError.message : 'System information failed');
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const details = useMemo<Detail[]>(() => {
    if (!systemInfo) {
      return [];
    }

    const nextDetails: Detail[] = [];
    if (systemInfo.platform) {
      nextDetails.push({
        label: 'Operating System',
        value: `${displayValue(systemInfo.platform.system)} ${displayValue(systemInfo.platform.release)}`,
      });
      nextDetails.push({ label: 'Architecture', value: displayValue(systemInfo.platform.machine) });
    }

    if (systemInfo.inference) {
      const inference = systemInfo.inference;
      nextDetails.push({ label: 'Inference Engine', value: displayEngine(inference.engine) });
      if (inference.acceleration) {
        nextDetails.push({ label: 'Acceleration', value: inference.acceleration });
      }
    }

    if (systemInfo.python) {
      nextDetails.push({ label: 'Python Runtime', value: displayValue(systemInfo.python.version) });
    }
    return nextDetails;
  }, [systemInfo]);

  const productName = branding.productName ?? 'Geist';
  const productVersion = branding.productVersion ?? systemInfo?.version;

  return (
    <section className="about-section" aria-labelledby="about-product-name">
      <header className="about-identity">
        <AboutAuroraBackground />
        <div className="about-lockup">
          {branding.logoUrl && (
            <img className="about-logo" src={branding.logoUrl} alt={`${productName} logo`} />
          )}
          <div className="about-identity-copy">
            <h2 id="about-product-name">
              {productName}
              {productVersion && <span className="about-product-version">v{productVersion}</span>}
            </h2>
            <p>
              Powered by Geist
              {systemInfo?.version && <span> v{systemInfo.version}</span>}
            </p>
          </div>
        </div>
      </header>

      <div className="about-machine">
        <header className="settings-section-header">
          <h3>This Machine</h3>
          <p>Runtime details reported by the active Geist service.</p>
        </header>

        {!systemInfo && !error && <div className="empty-state compact">Loading system information...</div>}
        {error && <div className="notice notice-error">{error}</div>}
        {systemInfo && (
          <>
            {details.length > 1 ? (
              <dl className="about-details">
                {details.map((detail) => (
                  <div className="about-detail" key={detail.label}>
                    <dt>{detail.label}</dt>
                    <dd>{detail.value}</dd>
                  </div>
                ))}
              </dl>
            ) : (
              <div className="notice">
                Restart Geist to load detailed machine and inference information.
              </div>
            )}
          </>
        )}
      </div>
    </section>
  );
}
