---
permalink: /
layout: homepage
title: "Penghui Yang"
excerpt: "Ph.D. student working on multimodal learning and large language models, with a growing interest in AI agents."
redirect_from: 
  - /about/
  - /about.html
---

<section class="home-hero" aria-labelledby="hero-title">
  <div class="home-hero__copy">
    <p class="home-eyebrow"><span></span>Hello, I’m</p>
    <h1 id="hero-title">Penghui Yang <span>杨鹏晖</span></h1>
    <p class="home-hero__role">Ph.D. Student · Multimodal AI Researcher · Exploring AI Agents</p>
    <p class="home-hero__intro">
      I study how multimodal foundation models learn, reason, and align, and I am also interested in
      AI agents that can perceive, reason, and act in complex environments. I am a Ph.D. student at
      <a href="https://www.shlab.org.cn/">Shanghai AI Lab</a>, jointly trained with the
      <a href="https://soai.sjtu.edu.cn/">School of Artificial Intelligence, Shanghai Jiao Tong University</a>,
      under the supervision of <a href="https://yuhangzang.github.io/">Yuhang Zang</a>.
    </p>
    <div class="home-tags" aria-label="Research interests">
      <span>Multimodal Learning</span>
      <span>Vision–Language Models</span>
      <span>Post-training Alignment</span>
      <span>AI Agents</span>
    </div>
    <div class="home-actions">
      <a class="home-button home-button--primary" href="/publications/">View my research <i class="fas fa-arrow-right" aria-hidden="true"></i></a>
      <a class="home-icon-button" href="mailto:yangpenghui1@pjlab.org.cn" aria-label="Email Penghui Yang"><i class="fas fa-envelope" aria-hidden="true"></i></a>
      <a class="home-icon-button" href="https://github.com/yph22" aria-label="Penghui Yang on GitHub"><i class="fab fa-github" aria-hidden="true"></i></a>
      <a class="home-icon-button" href="https://www.xiaohongshu.com/user/profile/641d022a0000000011020415" aria-label="Penghui Yang on Xiaohongshu">{% include xiaohongshu-icon.svg %}</a>
    </div>
  </div>

  <div class="home-portrait-wrap" aria-label="Portrait of Penghui Yang">
    <div class="home-portrait__accent home-portrait__accent--one"></div>
    <div class="home-portrait__accent home-portrait__accent--two"></div>
    <div class="home-portrait">
      <img src="/images/AI.png" alt="Penghui Yang" fetchpriority="high">
    </div>
    <div class="home-status"><span></span> Based in Shanghai, China</div>
  </div>
</section>

<section class="home-section" aria-labelledby="research-title">
  <div class="home-section__heading">
    <p class="home-kicker">Research focus</p>
    <h2 id="research-title">Building capable, efficient, and trustworthy multimodal systems.</h2>
    <p>My work sits at the intersection of representation learning, model efficiency, and alignment.</p>
  </div>
  <div class="research-grid">
    <article class="research-card">
      <span class="research-card__number">01</span>
      <i class="fas fa-layer-group" aria-hidden="true"></i>
      <h3>Multimodal Learning</h3>
      <p>Learning unified representations across vision, language, and diverse real-world signals.</p>
    </article>
    <article class="research-card">
      <span class="research-card__number">02</span>
      <i class="fas fa-eye" aria-hidden="true"></i>
      <h3>Vision–Language Models</h3>
      <p>Improving visual understanding through scalable pre-training and effective data design.</p>
    </article>
    <article class="research-card">
      <span class="research-card__number">03</span>
      <i class="fas fa-sliders" aria-hidden="true"></i>
      <h3>LLM Post-training/Alignment</h3>
      <p>Aligning foundation models with reliable rewards, preferences, and human intent.</p>
    </article>
  </div>
</section>

<section class="home-section home-feature" aria-labelledby="resopd-featured-title">
  <div class="home-feature__visual">
    <img src="/images/publications/resopd-overview.png" alt="ResOPD overview: sparse teacher feedback, gradient variance reduction, and reasoning performance" loading="lazy">
  </div>
  <div class="home-feature__copy">
    <p class="home-kicker">Featured work · 2026</p>
    <h2 id="resopd-featured-title">ResOPD</h2>
    <p class="home-feature__subtitle">Tail Residualization for Sparse On-Policy Distillation</p>
    <p>ResOPD computes the coarse gradient exactly and samples only the fine-tail residual, reducing gradient variance with the same sparse teacher feedback.</p>
    <div class="home-feature__meta">
      <span>arXiv Preprint</span>
      <span>On-Policy Distillation</span>
    </div>
    <div class="home-actions">
      <a class="home-text-link" href="https://arxiv.org/abs/2610.04882">Paper <i class="fas fa-arrow-up-right-from-square" aria-hidden="true"></i></a>
      <a class="home-text-link" href="https://github.com/InternLM/ResOPD">Code <i class="fab fa-github" aria-hidden="true"></i></a>
      <a class="home-text-link" href="https://huggingface.co/datasets/ygyjrc/ResOPD-deepmath">Dataset <i class="fas fa-database" aria-hidden="true"></i></a>
    </div>
  </div>
</section>

<section class="home-section home-feature" aria-labelledby="featured-title">
  <div class="home-feature__visual">
    <img src="/images/publications/caprlpp-comparison.png" alt="CapRL++ method comparison diagram" loading="lazy">
  </div>
  <div class="home-feature__copy">
    <p class="home-kicker">Featured work · 2026</p>
    <h2 id="featured-title">CapRL++</h2>
    <p class="home-feature__subtitle">A unified framework for dense visual captioning and multimodal post-training.</p>
    <p>This work explores a scalable approach to stronger perception and alignment in multimodal large language models.</p>
    <div class="home-feature__meta">
      <span>TPAMI Submission</span>
      <span>Multimodal LLMs</span>
    </div>
    <a class="home-text-link" href="https://arxiv.org/abs/2606.09393">Read the paper <i class="fas fa-arrow-up-right-from-square" aria-hidden="true"></i></a>
  </div>
</section>

<section id="projects" class="home-section home-projects" aria-labelledby="projects-title">
  <div class="home-section__heading">
    <p class="home-kicker">Open-source projects</p>
    <h2 id="projects-title">Projects</h2>
    <p>Multimodal models and tools that turn perception into decisions.</p>
  </div>
  <article class="home-feature home-project home-project--valen" aria-labelledby="valen-title">
    <div class="valen-stage" aria-label="Valen: text, image and video inputs to candidate probabilities">
      <div class="valen-stage__orbit" aria-hidden="true"></div>
      <div class="valen-stage__header"><span class="valen-stage__dot" aria-hidden="true"></span> MULTIMODAL DECISION MODEL</div>
      <div class="valen-stage__inputs" aria-hidden="true">
        <span><i class="fas fa-align-left"></i> Text</span>
        <span><i class="fas fa-image"></i> Image</span>
        <span><i class="fas fa-film"></i> Video</span>
      </div>
      <div class="valen-stage__connector" aria-hidden="true"></div>
      <div class="valen-stage__logo">
        <img src="{{ '/images/projects/valen-logo.svg' | relative_url }}" alt="Valen logo" width="2052" height="583" loading="lazy">
      </div>
      <p class="valen-stage__slogan">System One Model,<br><strong>now with vision.</strong></p>
      <div class="valen-stage__output" aria-hidden="true">
        <span>OUTPUT</span><strong>Candidate probabilities</strong>
        <div class="valen-stage__bars"><i></i><i></i><i></i></div>
      </div>
    </div>
    <div class="home-feature__copy">
      <p class="home-kicker">Perception → Decision</p>
      <h3 id="valen-title" class="home-project__title">Valen <span>万澜</span></h3>
      <p class="home-feature__subtitle">Multimodal perception. Structured decisions.</p>
      <p>Valen brings visual perception to System One decision-making. It evaluates text, images, and video against task instructions, returning probabilities over supplied candidates.</p>
      <ul class="home-project__highlights">
        <li><strong>Three model sizes.</strong> Explore the 0.8B, 2B, and 4B models.</li>
        <li><strong>Train and adapt.</strong> Includes data processing, SFT, experimental RLCD, inference, and evaluation tools.</li>
        <li><strong>Evaluate visual decisions.</strong> VisualDecisionBench covers both image and video tasks.</li>
      </ul>
      <div class="home-feature__meta" aria-label="Project details">
        <span>Text · Image · Video</span>
        <span>0.8B / 2B / 4B</span>
        <span>Apache 2.0 code</span>
      </div>
      <div class="home-actions">
        <a class="home-button home-button--primary" href="https://github.com/Liuziyu77/Valen"><i class="fab fa-github" aria-hidden="true"></i> Explore Valen <i class="fas fa-arrow-right" aria-hidden="true"></i></a>
        <a class="home-button" href="https://huggingface.co/Valen-Team">Models <i class="fas fa-arrow-up-right-from-square" aria-hidden="true"></i></a>
      </div>
      <div class="valen-resource-links">
        <a class="home-text-link" href="https://huggingface.co/spaces/yuhangzang/Valen-Preview-0923">Try preview demo <i class="fas fa-arrow-up-right-from-square" aria-hidden="true"></i></a>
        <a class="home-text-link" href="https://github.com/Liuziyu77/Valen/tree/main/docs">Documentation <i class="fas fa-arrow-right" aria-hidden="true"></i></a>
      </div>
    </div>
  </article>
</section>

<section class="home-section home-two-column">
  <div class="home-panel" aria-labelledby="news-title">
    <div class="home-panel__header">
      <div>
        <p class="home-kicker">Updates</p>
        <h2 id="news-title">Latest news</h2>
      </div>
    </div>
    <ol class="home-timeline home-timeline--news">
      <li>
        <time datetime="2026-10-04">Oct 2026</time>
        <p>Our new preprint <a href="https://arxiv.org/abs/2610.04882"><strong>ResOPD</strong></a> introduces tail residualization for sparse on-policy distillation.</p>
      </li>
      <li>
        <time datetime="2026-06">Jun 2026</time>
        <p>Received the <strong>Beijing Outstanding Graduate</strong> and <strong>Tsinghua Excellent Graduate</strong> awards.</p>
      </li>
      <li>
        <time datetime="2026-06">Jun 2026</time>
        <p>Our work <a href="https://arxiv.org/abs/2606.09393"><strong>CapRL++</strong></a> was submitted to TPAMI.</p>
      </li>
      <li>
        <time datetime="2025-10-21">Oct 2025</time>
        <p>Started my internship at <a href="https://www.shlab.org.cn/">Shanghai AI Lab</a></p>
      </li>
      <li>
        <time datetime="2025-10-10">Oct 2025</time>
        <p>Received the <strong>National Scholarship</strong>.</p>
      </li>
      <li>
        <time datetime="2025-05">May 2025</time>
        <p>Won Second Prize in Tsinghua’s AGI Competition for a <a href="https://github.com/BEEE22/Cleanroom-RAG">RAG project</a>.</p>
      </li>
    </ol>
  </div>

  <div class="home-panel" aria-labelledby="journey-title">
    <div class="home-panel__header">
      <div>
        <p class="home-kicker">Background</p>
        <h2 id="journey-title">Education & experience</h2>
      </div>
    </div>
    <ol class="home-timeline">
      <li>
        <time>2026 — Present</time>
        <h3>Ph.D. Student</h3>
        <p>Shanghai AI Lab × Shanghai Jiao Tong University</p>
      </li>
      <li>
        <time>2025 — Present</time>
        <h3>Research Intern</h3>
        <p>Shanghai Artificial Intelligence Laboratory</p>
      </li>
      <li>
        <time>2022 — 2026</time>
        <h3>B.Eng. · Rank 1st</h3>
        <p>Building Environment & Energy Engineering, Tsinghua University</p>
      </li>
    </ol>
  </div>
</section>

<section class="home-section home-honors" aria-labelledby="honors-title">
  <div class="home-section__heading home-section__heading--compact">
    <p class="home-kicker">Recognition</p>
    <h2 id="honors-title">Honors & awards</h2>
  </div>
  <div class="honors-list">
    <div><span>2026</span><strong>Beijing Outstanding Graduate</strong><small>Beijing Municipal Education Commission</small></div>
    <div><span>2026</span><strong>Tsinghua Excellent Graduate</strong><small>Tsinghua University</small></div>
    <div><span>2025</span><strong>National Scholarship</strong><small>Ministry of Education of China</small></div>
    <div><span>2025</span><strong>Second Prize · AGI Competition</strong><small>Tsinghua University</small></div>
    <div><span>2023–24</span><strong>Comprehensive Excellence Scholarship</strong><small>Top 10% · Tsinghua University</small></div>
  </div>
</section>

<section class="home-section home-personal" aria-label="Outside research">
  <p class="home-kicker">Beyond the lab</p>
  <h2>Good ideas often come after a workout—or with good coffee.</h2>
  <p>Outside research, you’ll find me playing basketball or billiards, training at the gym, and exploring cafés around the city.</p>
  <div class="home-personal__icons" aria-hidden="true"><span>🏀</span><span>🏋️</span><span>🎱</span><span>☕</span></div>
</section>
