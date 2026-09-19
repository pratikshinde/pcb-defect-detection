module.exports = {
  apps: [
    {
      name: 'pcb-aoi-backend',
      script: 'npx',
      args: 'tsx src/index.ts',
      instances: 1, // Start with 1 instance for CV stability
      autorestart: true,
      watch: false,
      max_memory_restart: '1G',
      env: {
        NODE_ENV: 'development',
      },
      env_production: {
        NODE_ENV: 'production',
        PORT: 3000,
        // CV Parameters (Tuned for production lighting)
        CV_ADAPTIVE_THRESH_BLOCK_SIZE: 11,
        CV_ADAPTIVE_THRESH_C: 2,
        CV_MORPH_KERNEL_SIZE: 3,
        CV_MIN_DEFECT_AREA_PX: 10
      }
    }
  ]
};
