import React, { useEffect, useRef, useState } from 'react';
import { useTheme } from '../contexts/ThemeContext';

// Medical Background Component that adds animated visualizations
const MedicalBackground = () => {
  const canvasRef = useRef(null);
  const { darkMode } = useTheme();
  const [currentVariant, setCurrentVariant] = useState('default');
  const [transitionProgress, setTransitionProgress] = useState(0);
  const [isTransitioning, setIsTransitioning] = useState(false);

  // List of all available variants
  const allVariants = ['default', 'dna', 'cells', 'network', 'blood', 'brain'];
  const TRANSITION_DURATION = 2000; // 2 seconds for transition
  const VARIANT_DURATION = 8000; // 8 seconds per variant

  // Special effect: Neural Network
  const drawNeuralNetwork = (ctx, canvas, time, colors) => {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(canvas.width, canvas.height) * 0.3;
    
    // Draw faint circular path
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.strokeStyle = `${colors[0]}20`;
    ctx.lineWidth = 1;
    ctx.stroke();
    
    // Draw neurons along the path
    const neuronCount = 10;
    const neurons = [];
    
    for (let i = 0; i < neuronCount; i++) {
      const angle = (i / neuronCount) * Math.PI * 2 + time;
      const x = centerX + Math.cos(angle) * radius;
      const y = centerY + Math.sin(angle) * radius;
      neurons.push({x, y});
      
      // Draw neuron
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fillStyle = colors[i % colors.length];
      ctx.fill();
    }
    
    // Draw connections between neurons
    for (let i = 0; i < neurons.length; i++) {
      for (let j = i + 1; j < neurons.length; j++) {
        ctx.beginPath();
        ctx.moveTo(neurons[i].x, neurons[i].y);
        ctx.lineTo(neurons[j].x, neurons[j].y);
        
        // Pulse effect along connections
        const pulsePos = (time * 2) % 1;
        
        const gradient = ctx.createLinearGradient(
          neurons[i].x, neurons[i].y, 
          neurons[j].x, neurons[j].y
        );
        
        gradient.addColorStop(Math.max(0, pulsePos - 0.1), `${colors[0]}10`);
        gradient.addColorStop(pulsePos, `${colors[0]}80`);
        gradient.addColorStop(Math.min(1, pulsePos + 0.1), `${colors[0]}10`);
        
        ctx.strokeStyle = gradient;
        ctx.lineWidth = 0.5;
        ctx.stroke();
      }
    }
  };
  
  // Special effect: DNA Helix
  const drawDNAHelix = (ctx, canvas, time, colors) => {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    const width = Math.min(canvas.width, canvas.height) * 0.6;
    const height = width * 0.4;
    
    // Draw the double helix
    const points = 20;
    const leftStrand = [];
    const rightStrand = [];
    
    for (let i = 0; i < points; i++) {
      const t = i / points;
      const angle = t * Math.PI * 4 + time;
      
      // Left strand
      const leftX = centerX + Math.cos(angle) * width / 2;
      const leftY = centerY + t * height - height / 2;
      leftStrand.push({x: leftX, y: leftY});
      
      // Right strand (opposite phase)
      const rightX = centerX - Math.cos(angle) * width / 2;
      const rightY = centerY + t * height - height / 2;
      rightStrand.push({x: rightX, y: rightY});
      
      // Draw connections between strands (base pairs)
      if (i % 2 === 0) {
        ctx.beginPath();
        ctx.moveTo(leftX, leftY);
        ctx.lineTo(rightX, rightY);
        ctx.strokeStyle = `${colors[i % colors.length]}40`;
        ctx.lineWidth = 1;
        ctx.stroke();
        
        // Draw nucleotides at each end
        ctx.font = '8px Arial';
        ctx.fillStyle = colors[i % colors.length];
        
        const baseA = i % 4 === 0 ? 'A' : i % 4 === 1 ? 'T' : i % 4 === 2 ? 'G' : 'C';
        const baseB = baseA === 'A' ? 'T' : baseA === 'T' ? 'A' : baseA === 'G' ? 'C' : 'G';
        
        ctx.fillText(baseA, leftX - 4, leftY + 4);
        ctx.fillText(baseB, rightX - 4, rightY + 4);
      }
    }
    
    // Draw the strands
    ctx.beginPath();
    ctx.moveTo(leftStrand[0].x, leftStrand[0].y);
    for (let i = 1; i < leftStrand.length; i++) {
      ctx.lineTo(leftStrand[i].x, leftStrand[i].y);
    }
    ctx.strokeStyle = `${colors[0]}80`;
    ctx.lineWidth = 2;
    ctx.stroke();
    
    ctx.beginPath();
    ctx.moveTo(rightStrand[0].x, rightStrand[0].y);
    for (let i = 1; i < rightStrand.length; i++) {
      ctx.lineTo(rightStrand[i].x, rightStrand[i].y);
    }
    ctx.strokeStyle = `${colors[1]}80`;
    ctx.lineWidth = 2;
    ctx.stroke();
  };
  
  // Special effect: Cell Division
  const drawCellDivision = (ctx, canvas, time, colors) => {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(canvas.width, canvas.height) * 0.15;
    
    // Draw main cell
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.fillStyle = `${colors[0]}20`;
    ctx.fill();
    ctx.strokeStyle = `${colors[0]}60`;
    ctx.lineWidth = 1;
    ctx.stroke();
    
    // Draw dividing cell effect
    const divisionPhase = (Math.sin(time) + 1) / 2; // 0 to 1
    
    if (divisionPhase > 0.3) {
      // Draw constriction in the middle
      const constrictionWidth = radius * (1 - (divisionPhase - 0.3) / 0.7 * 0.8);
      
      ctx.beginPath();
      ctx.ellipse(
        centerX, 
        centerY, 
        constrictionWidth, 
        radius, 
        0, 
        0, 
        Math.PI * 2
      );
      ctx.strokeStyle = `${colors[1]}70`;
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Draw chromosomes
      if (divisionPhase < 0.6) {
        const chromosomeCount = 5;
        for (let i = 0; i < chromosomeCount; i++) {
          const angle = (i / chromosomeCount) * Math.PI * 2;
          const distance = radius * 0.5;
          const x = centerX + Math.cos(angle) * distance;
          const y = centerY + Math.sin(angle) * distance;
          
          ctx.beginPath();
          ctx.moveTo(x - 5, y - 5);
          ctx.lineTo(x + 5, y + 5);
          ctx.moveTo(x - 5, y + 5);
          ctx.lineTo(x + 5, y - 5);
          ctx.strokeStyle = `${colors[2]}90`;
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      } else {
        // Draw two sets of chromosomes
        const chromosomeCount = 5;
        for (let i = 0; i < chromosomeCount; i++) {
          const angle = (i / chromosomeCount) * Math.PI * 2;
          const distance = radius * 0.3;
          
          // Left cell chromosomes
          const x1 = centerX - radius * 0.5 + Math.cos(angle) * distance;
          const y1 = centerY + Math.sin(angle) * distance;
          
          ctx.beginPath();
          ctx.moveTo(x1 - 3, y1 - 3);
          ctx.lineTo(x1 + 3, y1 + 3);
          ctx.moveTo(x1 - 3, y1 + 3);
          ctx.lineTo(x1 + 3, y1 - 3);
          ctx.strokeStyle = `${colors[2]}90`;
          ctx.lineWidth = 1;
          ctx.stroke();
          
          // Right cell chromosomes
          const x2 = centerX + radius * 0.5 + Math.cos(angle) * distance;
          const y2 = centerY + Math.sin(angle) * distance;
          
          ctx.beginPath();
          ctx.moveTo(x2 - 3, y2 - 3);
          ctx.lineTo(x2 + 3, y2 + 3);
          ctx.moveTo(x2 - 3, y2 + 3);
          ctx.lineTo(x2 + 3, y2 - 3);
          ctx.strokeStyle = `${colors[2]}90`;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
    } else {
      // Draw nucleus
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius * 0.5, 0, Math.PI * 2);
      ctx.strokeStyle = `${colors[1]}40`;
      ctx.fillStyle = `${colors[1]}20`;
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  };
  
  // Special effect: Pulse/Heartbeat
  const drawPulseEffect = (ctx, canvas, time, colors) => {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    // Draw ECG line
    const lineWidth = Math.min(canvas.width, canvas.height) * 0.6;
    const lineHeight = Math.min(canvas.width, canvas.height) * 0.1;
    
    const startX = centerX - lineWidth / 2;
    const endX = centerX + lineWidth / 2;
    
    // Draw the baseline
    ctx.beginPath();
    ctx.moveTo(startX, centerY);
    ctx.lineTo(endX, centerY);
    ctx.strokeStyle = `${colors[0]}30`;
    ctx.lineWidth = 1;
    ctx.stroke();
    
    // Draw animated heartbeat
    const heartbeatSpeed = 2;
    const t = (time * heartbeatSpeed) % 1; 
    
    if (t < 0.7) {
      // Draw ECG pattern
      const progress = t / 0.7; // 0 to 1 during active phase
      
      ctx.beginPath();
      ctx.moveTo(startX, centerY);
      
      // First segment - flat line
      const segment1End = startX + lineWidth * 0.3;
      ctx.lineTo(Math.min(segment1End, startX + progress * lineWidth), centerY);
      
      if (progress > 0.3) {
        // P wave
        const pWaveStart = startX + lineWidth * 0.3;
        const pWaveEnd = startX + lineWidth * 0.4;
        
        ctx.quadraticCurveTo(
          pWaveStart + (pWaveEnd - pWaveStart) / 2, 
          centerY - lineHeight * 0.2,
          Math.min(pWaveEnd, startX + progress * lineWidth), 
          centerY
        );
      }
      
      if (progress > 0.4) {
        // Flat before QRS
        const preQRSEnd = startX + lineWidth * 0.5;
        ctx.lineTo(Math.min(preQRSEnd, startX + progress * lineWidth), centerY);
      }
      
      if (progress > 0.5) {
        // QRS complex
        const qPeak = startX + lineWidth * 0.52;
        const rPeak = startX + lineWidth * 0.55;
        const sPeak = startX + lineWidth * 0.58;
        const qrsEnd = startX + lineWidth * 0.6;
        
        // Q dip
        if (progress > 0.52) {
          ctx.lineTo(qPeak, centerY + lineHeight * 0.2);
        }
        
        // R peak
        if (progress > 0.55) {
          ctx.lineTo(rPeak, centerY - lineHeight * 0.8);
        }
        
        // S dip
        if (progress > 0.58) {
          ctx.lineTo(sPeak, centerY + lineHeight * 0.4);
        }
        
        // End of QRS
        if (progress > 0.6) {
          ctx.lineTo(Math.min(qrsEnd, startX + progress * lineWidth), centerY);
        }
      }
      
      if (progress > 0.6) {
        // T wave
        const tWaveStart = startX + lineWidth * 0.65;
        const tWaveEnd = startX + lineWidth * 0.75;
        
        if (progress > 0.65) {
          ctx.lineTo(tWaveStart, centerY);
          
          if (progress > 0.75) {
            ctx.quadraticCurveTo(
              tWaveStart + (tWaveEnd - tWaveStart) / 2, 
              centerY - lineHeight * 0.3,
              Math.min(tWaveEnd, startX + progress * lineWidth), 
              centerY
            );
          }
        }
      }
      
      if (progress > 0.75) {
        // Final flat line
        ctx.lineTo(startX + progress * lineWidth, centerY);
      }
      
      ctx.strokeStyle = colors[0];
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Animate a pulse ring at QRS peak
      if (progress > 0.55 && progress < 0.9) {
        const pulseProgress = (progress - 0.55) / 0.35; // 0 to 1
        const pulseRadius = pulseProgress * lineHeight * 2;
        
        ctx.beginPath();
        ctx.arc(startX + lineWidth * 0.55, centerY - lineHeight * 0.8, pulseRadius, 0, Math.PI * 2);
        ctx.fillStyle = `${colors[0]}${Math.floor((1 - pulseProgress) * 40).toString(16).padStart(2, '0')}`;
        ctx.fill();
      }
    }
  };
  
  // Special effect: Blood Flow
  const drawBloodFlow = (ctx, canvas, time, colors) => {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(canvas.width, canvas.height) * 0.3;
    
    // Draw blood vessels
    const vesselCount = 8;
    for (let i = 0; i < vesselCount; i++) {
      const angle = (i / vesselCount) * Math.PI * 2;
      const startX = centerX + Math.cos(angle) * radius * 0.3;
      const startY = centerY + Math.sin(angle) * radius * 0.3;
      const endX = centerX + Math.cos(angle) * radius;
      const endY = centerY + Math.sin(angle) * radius;
      
      // Draw vessel
      ctx.beginPath();
      ctx.moveTo(startX, startY);
      ctx.lineTo(endX, endY);
      ctx.strokeStyle = `${colors[0]}30`;
      ctx.lineWidth = 3;
      ctx.stroke();
      
      // Draw blood cells flowing
      const cellCount = 5;
      for (let j = 0; j < cellCount; j++) {
        const progress = ((time * 2 + j / cellCount) % 1);
        const x = startX + (endX - startX) * progress;
        const y = startY + (endY - startY) * progress;
        
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fillStyle = colors[1];
        ctx.fill();
      }
    }
  };
  
  // Special effect: Brain Activity
  const drawBrainActivity = (ctx, canvas, time, colors) => {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(canvas.width, canvas.height) * 0.25;
    
    // Draw brain outline
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.strokeStyle = `${colors[0]}40`;
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Draw neural activity
    const pointCount = 20;
    for (let i = 0; i < pointCount; i++) {
      const angle = (i / pointCount) * Math.PI * 2;
      const distance = radius * (0.7 + Math.sin(time * 3 + i) * 0.1);
      const x = centerX + Math.cos(angle) * distance;
      const y = centerY + Math.sin(angle) * distance;
      
      // Draw activity point
      ctx.beginPath();
      ctx.arc(x, y, 2 + Math.sin(time * 5 + i) * 2, 0, Math.PI * 2);
      ctx.fillStyle = `${colors[1]}${Math.floor((Math.sin(time * 3 + i) + 1) * 40 + 20).toString(16).padStart(2, '0')}`;
      ctx.fill();
      
      // Draw connections
      for (let j = i + 1; j < pointCount; j++) {
        if (Math.random() > 0.7) {
          const angle2 = (j / pointCount) * Math.PI * 2;
          const distance2 = radius * (0.7 + Math.sin(time * 3 + j) * 0.1);
          const x2 = centerX + Math.cos(angle2) * distance2;
          const y2 = centerY + Math.sin(angle2) * distance2;
          
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(x2, y2);
          ctx.strokeStyle = `${colors[2]}20`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
  };
  
  // Different animation variants
  const variants = {
    default: {
      particleCount: 40,
      connectionDistance: 150,
      speed: 0.5,
      size: 3,
      colors: {
        light: ['#3b82f6', '#60a5fa', '#93c5fd', '#1e40af'],
        dark: ['#1e40af', '#3b82f6', '#60a5fa', '#93c5fd'],
      },
      symbols: ['●', '▲', '■'], // Basic shapes
      drawSpecial: drawNeuralNetwork
    },
    dna: {
      particleCount: 60,
      connectionDistance: 100,
      speed: 0.3,
      size: 2,
      colors: {
        light: ['#8b5cf6', '#a78bfa', '#c4b5fd', '#7c3aed'],
        dark: ['#7c3aed', '#8b5cf6', '#a78bfa', '#c4b5fd'],
      },
      symbols: ['A', 'T', 'G', 'C'], // DNA nucleotides
      drawSpecial: drawDNAHelix
    },
    cells: {
      particleCount: 30,
      connectionDistance: 200,
      speed: 0.7,
      size: 4,
      colors: {
        light: ['#10b981', '#34d399', '#6ee7b7', '#059669'],
        dark: ['#059669', '#10b981', '#34d399', '#6ee7b7'],
      },
      symbols: ['⬤', '⚕', '+', '◉'],  // Cell & medical symbols
      drawSpecial: drawCellDivision
    },
    network: {
      particleCount: 80,
      connectionDistance: 120,
      speed: 0.4,
      size: 2,
      colors: {
        light: ['#ef4444', '#f87171', '#fca5a5', '#dc2626'],
        dark: ['#dc2626', '#ef4444', '#f87171', '#fca5a5'],
      },
      symbols: ['❤', '🫁', '🧠'], // Organ symbols
      drawSpecial: drawPulseEffect
    },
    blood: {
      particleCount: 50,
      connectionDistance: 180,
      speed: 0.6,
      size: 3,
      colors: {
        light: ['#dc2626', '#ef4444', '#f87171', '#b91c1c'],
        dark: ['#b91c1c', '#dc2626', '#ef4444', '#f87171'],
      },
      symbols: ['🩸', '💉', '❤️'],
      drawSpecial: drawBloodFlow
    },
    brain: {
      particleCount: 70,
      connectionDistance: 140,
      speed: 0.5,
      size: 2,
      colors: {
        light: ['#7c3aed', '#8b5cf6', '#a78bfa', '#6d28d9'],
        dark: ['#6d28d9', '#7c3aed', '#8b5cf6', '#a78bfa'],
      },
      symbols: ['🧠', '⚡', '💭'],
      drawSpecial: drawBrainActivity
    }
  };

  // Add transition effect
  const drawTransition = (ctx, canvas, progress, fromColors, toColors) => {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(canvas.width, canvas.height) * 0.4;
    
    // Draw transition circle
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius * progress, 0, Math.PI * 2);
    ctx.strokeStyle = `${toColors[0]}${Math.floor(progress * 40).toString(16).padStart(2, '0')}`;
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Draw particles
    const particleCount = 20;
    for (let i = 0; i < particleCount; i++) {
      const angle = (i / particleCount) * Math.PI * 2;
      const distance = radius * progress;
      const x = centerX + Math.cos(angle) * distance;
      const y = centerY + Math.sin(angle) * distance;
      
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fillStyle = `${toColors[1]}${Math.floor(progress * 60).toString(16).padStart(2, '0')}`;
      ctx.fill();
    }
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let animationFrameId;
    let particles = [];
    let time = 0;
    let variantIndex = 0;
    let lastVariantChange = Date.now();
    let transitionStartTime = 0;
    
    // Set canvas dimensions
    const handleResize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    
    window.addEventListener('resize', handleResize);
    handleResize();
    
    // Initialize particles
    const initializeParticles = (variantConfig) => {
      const colors = darkMode ? variantConfig.colors.dark : variantConfig.colors.light;
      particles = [];
      
      for (let i = 0; i < variantConfig.particleCount; i++) {
        particles.push({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          vx: (Math.random() - 0.5) * variantConfig.speed,
          vy: (Math.random() - 0.5) * variantConfig.speed,
          radius: Math.random() * variantConfig.size + 1,
          color: colors[Math.floor(Math.random() * colors.length)],
          symbol: variantConfig.symbols[Math.floor(Math.random() * variantConfig.symbols.length)],
          rotation: Math.random() * Math.PI * 2,
          rotationSpeed: (Math.random() - 0.5) * 0.02,
          pulsePhase: Math.random() * Math.PI * 2,
          scale: 1 + Math.random() * 0.5
        });
      }
    };
    
    // Animation function
    const animate = () => {
      const currentTime = Date.now();
      time += 0.01;
      
      // Check if it's time to change variants
      if (currentTime - lastVariantChange >= VARIANT_DURATION && !isTransitioning) {
        setIsTransitioning(true);
        transitionStartTime = currentTime;
        variantIndex = (variantIndex + 1) % allVariants.length;
        setCurrentVariant(allVariants[variantIndex]);
      }
      
      // Handle transition
      if (isTransitioning) {
        const transitionTime = currentTime - transitionStartTime;
        const progress = Math.min(transitionTime / TRANSITION_DURATION, 1);
        setTransitionProgress(progress);
        
        if (progress >= 1) {
          setIsTransitioning(false);
          lastVariantChange = currentTime;
          initializeParticles(variants[allVariants[variantIndex]]);
        }
      }
      
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Get current and next variant configs
      const currentConfig = variants[allVariants[variantIndex]];
      const nextConfig = variants[allVariants[(variantIndex + 1) % allVariants.length]];
      const currentColors = darkMode ? currentConfig.colors.dark : currentConfig.colors.light;
      const nextColors = darkMode ? nextConfig.colors.dark : nextConfig.colors.light;
      
      // Draw current visualization
      if (currentConfig.drawSpecial) {
        currentConfig.drawSpecial(ctx, canvas, time, currentColors, darkMode);
      }
      
      // Draw transition effect if transitioning
      if (isTransitioning) {
        drawTransition(ctx, canvas, transitionProgress, currentColors, nextColors);
      }
      
      // Draw and update particles
      particles.forEach(particle => {
        // Move particle
        particle.x += particle.vx;
        particle.y += particle.vy;
        particle.rotation += particle.rotationSpeed;
        
        // Boundary check
        if (particle.x < 0 || particle.x > canvas.width) particle.vx *= -1;
        if (particle.y < 0 || particle.y > canvas.height) particle.vy *= -1;
        
        // Draw particle - either as circle or symbol
        if (Math.random() > 0.7) {
          // Draw as circle
          ctx.beginPath();
          ctx.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
          ctx.fillStyle = particle.color;
          ctx.fill();
        } else {
          // Draw as symbol
          ctx.save();
          ctx.translate(particle.x, particle.y);
          ctx.rotate(particle.rotation);
          ctx.font = `${particle.radius * 4 * particle.scale}px Arial`;
          ctx.fillStyle = particle.color;
          ctx.fillText(particle.symbol, 0, 0);
          ctx.restore();
        }
        
        // Connect particles within range
        if (currentVariant === 'dna') {
          drawDNAEffect(particle, particles, ctx, currentConfig, time);
        } else {
          drawConnections(particle, particles, ctx, currentConfig);
        }
      });
      
      animationFrameId = requestAnimationFrame(animate);
    };
    
    // Draw connections between nearby particles
    const drawConnections = (particle, particles, ctx, config) => {
      particles.forEach(other => {
        if (particle === other) return;
        
        const dx = particle.x - other.x;
        const dy = particle.y - other.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < config.connectionDistance) {
          ctx.beginPath();
          ctx.moveTo(particle.x, particle.y);
          ctx.lineTo(other.x, other.y);
          ctx.strokeStyle = `${particle.color}${Math.floor((1 - distance / config.connectionDistance) * 255).toString(16).padStart(2, '0')}`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      });
    };
    
    // Special DNA-like effect
    const drawDNAEffect = (particle, particles, ctx, config, time) => {
      const wavelength = 200;
      const amplitude = 50;
      
      particles.forEach(other => {
        if (particle === other) return;
        
        const dx = particle.x - other.x;
        const dy = particle.y - other.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < config.connectionDistance) {
          const wave1 = Math.sin((particle.x + time * 100) / wavelength) * amplitude;
          const wave2 = Math.sin((other.x + time * 100) / wavelength) * amplitude;
          
          ctx.beginPath();
          ctx.moveTo(particle.x, particle.y + wave1);
          ctx.lineTo(other.x, other.y + wave2);
          ctx.strokeStyle = `${particle.color}${Math.floor((1 - distance / config.connectionDistance) * 255).toString(16).padStart(2, '0')}`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      });
    };
    
    // Initialize first variant
    initializeParticles(variants[allVariants[0]]);
    animate();
    
    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);
    };
  }, [darkMode]);

  return (
    <canvas
      ref={canvasRef}
      className="fixed top-0 left-0 w-full h-full -z-10 pointer-events-none"
      style={{ opacity: 0.15 }}
    />
  );
};

export default MedicalBackground; 