// ─────────────────────────────────────────────────────────────────────────────
//  Cell Biology System
//  
//  This module encapsulates all cell-related simulation logic:
//  - Cell structure (membrane, flagella, injectors)
//  - Cell-specific physics (spring constraints, beating)
//  - Cell lifecycle (absorption, reproduction)
//
//  Separated from core particle physics to keep systems modular.
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────────────────────────────────────
const MEMBRANE_SIZE   = 40;          // particles per cell membrane
const FLAGELLA_COUNT  = 8;           // flagella per cell (evenly spaced)
const FLAGELLA_LENGTH = 7;           // particles per flagellum
const INJECTOR_COUNT  = 4;           // injector particles per cell (red, absorb liquid)
const CELL_LINK_BYTES = 32;          // prev(u32) + next(u32) + restDist(f32) + restAngle(f32) + cellStart(u32) + cellSize(u32) + chainIdx(u32) + ptype(u32)

// ─────────────────────────────────────────────────────────────────────────────
//  Particle types (ptype field in CellLink)
// ─────────────────────────────────────────────────────────────────────────────
const PTYPE = {
  MEMBRANE:  1,   // Cell membrane structure
  FLAGELLA:  2,   // Flagella chain
  INJECTOR:  3,   // Injection/absorption organ
  ABSORBED:  4,   // Absorbed food particle (still moves with cell)
  FOOD:      5,   // Free food particle
};

// ─────────────────────────────────────────────────────────────────────────────
//  CellLink struct definition (WGSL compatible)
// ─────────────────────────────────────────────────────────────────────────────
/*
struct CellLink {
  prev : u32,        // Index of previous particle in chain (0xFFFFFFFF = none)
  next : u32,        // Index of next particle in chain
  restDist : f32,    // Rest distance for spring constraints
  restAngle : f32,   // Rest angle for bending resistance
  cellStart : u32,   // Index of first particle in this cell
  cellSize : u32,    // Total number of particles in this cell
  chainIdx : u32,    // Position in chain (for flagella coordination)
  ptype : u32,       // Particle type (1=membrane, 2=flagella, 3=injector, 4=absorbed, 5=food)
}
*/

// ─────────────────────────────────────────────────────────────────────────────
//  Add a cell to the simulation
// ─────────────────────────────────────────────────────────────────────────────
function addCell(cx, cy, device, particleBuf, cellLinkBuf, particleCount, MAX_PARTICLES, sim) {
  const N  = MEMBRANE_SIZE;
  const FL = FLAGELLA_LENGTH;
  const FC = FLAGELLA_COUNT;
  const totalParts = N + FC * FL;

  const memRestDist = sim.particleRadius * 2;   // one diameter apart
  const R = memRestDist / (2 * Math.sin(Math.PI / N));

  if (particleCount + totalParts > MAX_PARTICLES) {
    return { success: false, message: `Need ${totalParts} free slots (have ${MAX_PARTICLES - particleCount}).` };
  }

  const baseIdx = particleCount;

  // ── Membrane particles ─────────────────────────────────────────────
  const flagSpacing = Math.floor(N / FC);
  const anchorIndices = new Set();
  for (let f = 0; f < FC; f++) {
    anchorIndices.add(f * flagSpacing);
  }
  const injSpacing = Math.floor(N / INJECTOR_COUNT);
  const injectorIndices = new Set();
  for (let ii = 0; ii < INJECTOR_COUNT; ii++) {
    injectorIndices.add(ii * injSpacing + 2);  // offset to avoid flagella anchors
  }

  const memF32 = new Float32Array(N * 8);
  for (let i = 0; i < N; i++) {
    const angle = (i / N) * 2 * Math.PI;
    const b = i * 8;
    memF32[b + 0] = cx + R * Math.cos(angle);
    memF32[b + 1] = cy + R * Math.sin(angle);
    memF32[b + 2] = 0;
    memF32[b + 3] = 0;
    if (anchorIndices.has(i)) {
      memF32[b + 4] = 1.0;   // orange (flagella anchor)
      memF32[b + 5] = 0.55;
      memF32[b + 6] = 0.1;
    } else if (injectorIndices.has(i)) {
      memF32[b + 4] = 1.0;   // red (injector)
      memF32[b + 5] = 0.1;
      memF32[b + 6] = 0.1;
    } else {
      memF32[b + 4] = 0.15;  // green
      memF32[b + 5] = 0.85;
      memF32[b + 6] = 0.35;
    }
    memF32[b + 7] = 1.0;   // color.a = 1.0 = full radius
  }
  device.queue.writeBuffer(particleBuf, baseIdx * 32, memF32.buffer);

  // ── Membrane cell-link data ────────────────────────────────────────
  const restAngle = Math.PI * (N - 2) / N;
  const memLinkAB = new ArrayBuffer(N * CELL_LINK_BYTES);
  const memLinkDV = new DataView(memLinkAB);
  for (let i = 0; i < N; i++) {
    const off = i * CELL_LINK_BYTES;
    memLinkDV.setUint32 (off + 0,  baseIdx + ((i - 1 + N) % N), true); // prev
    memLinkDV.setUint32 (off + 4,  baseIdx + ((i + 1)     % N), true); // next
    memLinkDV.setFloat32(off + 8,  memRestDist,  true);                // restDist
    memLinkDV.setFloat32(off + 12, restAngle, true);                   // restAngle
    memLinkDV.setUint32 (off + 16, baseIdx, true);                     // cellStart
    memLinkDV.setUint32 (off + 20, N, true);                           // cellSize
    memLinkDV.setUint32 (off + 24, 0, true);                           // chainIdx (unused)
    memLinkDV.setUint32 (off + 28, injectorIndices.has(i) ? PTYPE.INJECTOR : PTYPE.MEMBRANE, true);
  }
  device.queue.writeBuffer(cellLinkBuf, baseIdx * CELL_LINK_BYTES, memLinkAB);

  // ── Flagella particles + links ─────────────────────────────────────
  const flagParticles = new Float32Array(FC * FL * 8);
  const flagLinkAB    = new ArrayBuffer(FC * FL * CELL_LINK_BYTES);
  const flagLinkDV    = new DataView(flagLinkAB);
  const flagRestDist  = sim.particleRadius;  // one small-particle diameter
  const rootDist      = sim.particleRadius * 1.5;  // half big + half small diam

  for (let f = 0; f < FC; f++) {
    const memI  = f * flagSpacing;               // which membrane particle
    const angle = (memI / N) * 2 * Math.PI;      // outward direction
    const dirX  = Math.cos(angle);
    const dirY  = Math.sin(angle);

    for (let j = 0; j < FL; j++) {
      const gi = f * FL + j;   // index within flagella block
      const globalIdx = baseIdx + N + gi;

      // Position extending outward from membrane surface
      const dist = rootDist + j * flagRestDist;
      const b = gi * 8;
      flagParticles[b + 0] = cx + (R + dist) * dirX;
      flagParticles[b + 1] = cy + (R + dist) * dirY;
      flagParticles[b + 2] = 0;
      flagParticles[b + 3] = 0;
      flagParticles[b + 4] = 1.0;   // orange
      flagParticles[b + 5] = 0.55;
      flagParticles[b + 6] = 0.1;
      flagParticles[b + 7] = 0.5;   // color.a = 0.5 = half radius

      // CellLink data
      const off = gi * CELL_LINK_BYTES;
      const prevIdx = (j === 0) ? (baseIdx + memI) : (globalIdx - 1);
      const nextIdx = (j === FL - 1) ? 0xFFFFFFFF : (globalIdx + 1);
      const rd = (j === 0) ? rootDist : flagRestDist;

      flagLinkDV.setUint32 (off + 0,  prevIdx, true);       // prev
      flagLinkDV.setUint32 (off + 4,  nextIdx, true);       // next
      flagLinkDV.setFloat32(off + 8,  rd, true);             // restDist
      flagLinkDV.setFloat32(off + 12, 0, true);              // restAngle (unused)
      flagLinkDV.setUint32 (off + 16, baseIdx, true);        // cellStart
      flagLinkDV.setUint32 (off + 20, N, true);              // cellSize
      flagLinkDV.setUint32 (off + 24, j, true);              // chainIdx
      flagLinkDV.setUint32 (off + 28, PTYPE.FLAGELLA, true);
    }
  }
  device.queue.writeBuffer(particleBuf,  (baseIdx + N) * 32,  flagParticles.buffer);
  device.queue.writeBuffer(cellLinkBuf,  (baseIdx + N) * CELL_LINK_BYTES, flagLinkAB);

  return { 
    success: true, 
    newParticleCount: particleCount + totalParts,
    message: `Cell added! Particles: ${particleCount + totalParts}`
  };
}

// ─────────────────────────────────────────────────────────────────────────────
//  Initialize cell link buffer (fill with sentinels)
// ─────────────────────────────────────────────────────────────────────────────
function initializeCellLinkBuffer(device, cellLinkBuf, MAX_PARTICLES) {
  const sentinel = new Uint8Array(MAX_PARTICLES * CELL_LINK_BYTES);
  sentinel.fill(0xFF);
  device.queue.writeBuffer(cellLinkBuf, 0, sentinel);
}

// Export for use in HTML
window.cellSystem = {
  MEMBRANE_SIZE,
  FLAGELLA_COUNT,
  FLAGELLA_LENGTH,
  INJECTOR_COUNT,
  CELL_LINK_BYTES,
  PTYPE,
  addCell,
  initializeCellLinkBuffer,
};
