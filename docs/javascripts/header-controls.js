(() => {
  const initializeShellScrollTracking = () => {
    const header = document.querySelector(".md-header");
    if (!(header instanceof HTMLElement)) return;

    let updateScheduled = false;
    const updateShellScrollOffset = () => {
      updateScheduled = false;
      const offset = Math.min(window.scrollY, header.offsetHeight);
      document.documentElement.style.setProperty("--vh-shell-scroll-offset", `${offset}px`);
    };
    const scheduleShellScrollUpdate = () => {
      if (updateScheduled) return;
      updateScheduled = true;
      requestAnimationFrame(updateShellScrollOffset);
    };

    window.addEventListener("scroll", scheduleShellScrollUpdate, { passive: true });
    window.addEventListener("resize", scheduleShellScrollUpdate, { passive: true });
    updateShellScrollOffset();
  };

  const initializePrimaryNavigationControl = () => {
    const navigation = document.querySelector(".md-sidebar--primary");
    if (!(navigation instanceof HTMLElement)) return;

    navigation.querySelectorAll("label.md-nav__link[for]").forEach((label, index) => {
      if (!(label instanceof HTMLLabelElement)) return;
      const toggle = document.getElementById(label.htmlFor);
      const panel = Array.from(toggle?.parentElement?.children || [])
        .find((element) => element.classList.contains("md-nav"));
      if (!(toggle instanceof HTMLInputElement) || !(panel instanceof HTMLElement)) return;

      if (!panel.id) panel.id = `${toggle.id}_panel`;
      const button = document.createElement("button");
      Array.from(label.attributes).forEach((attribute) => {
        if (attribute.name !== "for" && attribute.name !== "role" && attribute.name !== "tabindex") {
          button.setAttribute(attribute.name, attribute.value);
        }
      });
      button.type = "button";
      button.innerHTML = label.innerHTML;
      button.dataset.vhNavToggle = toggle.id;
      button.setAttribute("aria-controls", panel.id);
      label.replaceWith(button);
      const sectionName = button.textContent?.trim().replace(/\s+/g, " ") || "Untitled";
      panel.setAttribute("aria-label", `Navigation section ${index + 1}: ${sectionName}`);
      panel.removeAttribute("aria-labelledby");
      if (panel.classList.contains("md-nav--secondary")) {
        panel.querySelectorAll("nav.md-nav").forEach((nestedPanel, nestedIndex) => {
          if (!(nestedPanel instanceof HTMLElement)) return;
          const labelledBy = nestedPanel.getAttribute("aria-labelledby");
          const labelledElement = labelledBy ? document.getElementById(labelledBy) : null;
          const nestedName = nestedPanel.getAttribute("aria-label")
            || labelledElement?.textContent?.trim().replace(/\s+/g, " ")
            || "Untitled";
          nestedPanel.setAttribute(
            "aria-label",
            `${sectionName} subsection ${nestedIndex + 1}: ${nestedName}`,
          );
          nestedPanel.removeAttribute("aria-labelledby");
        });
      }
      const synchronizeExpandedState = () => {
        button.setAttribute("aria-expanded", String(toggle.checked));
      };

      toggle.addEventListener("change", synchronizeExpandedState);
      button.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopImmediatePropagation();
        toggle.checked = !toggle.checked;
        toggle.dispatchEvent(new Event("change", { bubbles: true }));
      }, true);
      synchronizeExpandedState();
    });
  };

  const initializeTableOfContentsTracking = () => {
    const tableOfContents = document.querySelector(".md-sidebar--secondary");
    if (!(tableOfContents instanceof HTMLElement)) return;

    tableOfContents.addEventListener("click", (event) => {
      const link = event.target instanceof Element
        ? event.target.closest('.md-nav__link[href^="#"]')
        : null;
      if (!(link instanceof HTMLAnchorElement) || !link.hash) return;

      let settleTimer;
      const preserveSettledAnchor = () => {
        if (window.location.hash !== link.hash) {
          window.history.replaceState(window.history.state, "", link.hash);
        }
        tableOfContents.querySelectorAll('.md-nav__link[href^="#"]').forEach((trackedLink) => {
          trackedLink.classList.toggle("md-nav__link--active", trackedLink === link);
        });
      };
      const scheduleSettledAnchor = () => {
        window.clearTimeout(settleTimer);
        settleTimer = window.setTimeout(() => {
          window.removeEventListener("scroll", scheduleSettledAnchor);
          preserveSettledAnchor();
          window.setTimeout(preserveSettledAnchor, 500);
        }, 150);
      };

      window.addEventListener("scroll", scheduleSettledAnchor, { passive: true });
      scheduleSettledAnchor();
    });
    tableOfContents.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      const link = event.target instanceof Element
        ? event.target.closest('.md-nav__link[href^="#"]')
        : null;
      if (!(link instanceof HTMLAnchorElement) || !link.hash) return;
      event.preventDefault();
      link.click();
      link.focus({ preventScroll: true });
    });
  };

  const initializeVersionControl = () => {
    const control = document.querySelector("[data-vh-version-control]");
    if (!(control instanceof HTMLDetailsElement)) return;

    const summary = control.querySelector("summary");
    if (!(summary instanceof HTMLElement)) return;

    const synchronizeExpandedState = () => {
      summary.setAttribute("aria-expanded", String(control.open));
    };

    control.addEventListener("toggle", synchronizeExpandedState);
    control.addEventListener("keydown", (event) => {
      if (event.target === summary && (event.key === "Enter" || event.key === " ")) {
        event.preventDefault();
        control.open = !control.open;
        return;
      }
      if (event.key !== "Escape" || !control.open) return;
      control.open = false;
      summary.focus();
    });
    document.addEventListener("pointerdown", (event) => {
      if (control.open && !control.contains(event.target)) control.open = false;
    });
    synchronizeExpandedState();
  };

  const initializeSearchControl = () => {
    const checkbox = document.querySelector("#__search");
    const trigger = document.querySelector("[data-vh-search-trigger]");
    const search = document.querySelector(".md-search");
    const input = search?.querySelector(".md-search__input");
    const output = search?.querySelector(".md-search__output");
    const scrollwrap = output?.querySelector(".md-search__scrollwrap");
    const primaryShortcut = search?.querySelector("[data-vh-search-shortcut-primary]");
    const mobileViewport = window.matchMedia("(max-width: 59.984375em)");
    if (!(checkbox instanceof HTMLInputElement) || !(trigger instanceof HTMLElement)
        || !(search instanceof HTMLElement) || !(input instanceof HTMLInputElement)
        || !(output instanceof HTMLElement) || !(scrollwrap instanceof HTMLElement)) return;

    if (primaryShortcut instanceof HTMLElement) {
      const applePlatform = /Mac|iPhone|iPad/.test(navigator.platform);
      primaryShortcut.textContent = applePlatform ? "⌘K" : "Ctrl K";
    }

    let restoreFocus = false;
    const focusClosedSearchTarget = () => {
      const closeFocusTarget = mobileViewport.matches ? trigger : input;
      requestAnimationFrame(() => requestAnimationFrame(() => {
        if (!restoreFocus) return;
        closeFocusTarget.focus();
      }));
    };
    const synchronizeExpandedState = () => {
      const expanded = checkbox.checked;
      trigger.setAttribute("aria-expanded", String(expanded));
      document.body.classList.toggle("vh-search-open", expanded);
      input.tabIndex = expanded || !mobileViewport.matches ? 0 : -1;
      output.setAttribute("aria-hidden", String(!expanded));
      output.inert = !expanded;
      scrollwrap.tabIndex = expanded ? 0 : -1;
      if (expanded) {
        restoreFocus = true;
      } else if (restoreFocus) {
        focusClosedSearchTarget();
      }
    };
    const openSearch = () => {
      restoreFocus = true;
      if (!checkbox.checked) {
        checkbox.checked = true;
        checkbox.dispatchEvent(new Event("change", { bubbles: true }));
      }
      input.focus({ preventScroll: true });
    };
    const closeSearch = () => {
      restoreFocus = true;
      if (checkbox.checked) {
        checkbox.checked = false;
        checkbox.dispatchEvent(new Event("change", { bubbles: true }));
      } else {
        focusClosedSearchTarget();
      }
    };

    checkbox.addEventListener("change", synchronizeExpandedState);
    trigger.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopImmediatePropagation();
      openSearch();
    }, true);
    mobileViewport.addEventListener("change", synchronizeExpandedState);
    document.addEventListener("keydown", (event) => {
      if (event.key === "Tab" && document.activeElement === input) {
        restoreFocus = false;
        return;
      }
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        openSearch();
        return;
      }
      if (event.key === "Escape" && checkbox.checked) {
        event.preventDefault();
        event.stopImmediatePropagation();
        closeSearch();
      }
    }, true);
    synchronizeExpandedState();
  };

  const initializeLanguageControl = () => {
    const select = document.querySelector("[data-vh-language-select]");
    if (!(select instanceof HTMLSelectElement)) return;

    const paletteTransferKey = "vh-language-palette-transfer";
    try {
      const pendingPalette = sessionStorage.getItem(paletteTransferKey);
      sessionStorage.removeItem(paletteTransferKey);
      const pendingInput = document.querySelector(
        `input[data-md-color-scheme="${pendingPalette || ""}"]`
      );
      if (pendingInput instanceof HTMLInputElement && !pendingInput.checked) {
        pendingInput.checked = true;
        pendingInput.dispatchEvent(new Event("change", { bubbles: true }));
      }
    } catch (error) {
      console.debug("Language palette transfer is unavailable", error);
    }

    select.addEventListener("keydown", (event) => {
      if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      const direction = event.key === "ArrowDown" ? 1 : event.key === "ArrowUp" ? -1 : 0;
      if (!direction) return;
      const nextIndex = Math.max(0, Math.min(select.options.length - 1, select.selectedIndex + direction));
      if (nextIndex === select.selectedIndex) return;
      event.preventDefault();
      select.selectedIndex = nextIndex;
      select.dispatchEvent(new Event("change", { bubbles: true }));
    });
    select.addEventListener("change", () => {
      try {
        sessionStorage.setItem(paletteTransferKey, document.body.dataset.mdColorScheme || "default");
      } catch (error) {
        console.debug("Language palette transfer is unavailable", error);
      }
      if (select.value) window.location.assign(select.value);
    });
  };

  const initializeThemeControl = () => {
    const control = document.querySelector('[data-vh-header-control="theme"]');
    if (!(control instanceof HTMLElement)) return;

    const focusVisibleToggle = () => {
      requestAnimationFrame(() => requestAnimationFrame(() => {
        const visibleToggle = control.querySelector("[data-vh-theme-toggle]:not([hidden])");
        if (visibleToggle instanceof HTMLElement) visibleToggle.focus();
      }));
    };
    control.addEventListener("click", (event) => {
      const toggle = event.target instanceof Element
        ? event.target.closest("[data-vh-theme-toggle]")
        : null;
      if (!(toggle instanceof HTMLButtonElement)) return;
      const target = document.getElementById(toggle.dataset.vhThemeTarget || "");
      if (!(target instanceof HTMLInputElement)) return;
      target.checked = true;
      target.dispatchEvent(new Event("change", { bubbles: true }));
      focusVisibleToggle();
    });
    control.addEventListener("change", focusVisibleToggle);
  };

  const initializeCodeBlockLandmarks = () => {
    document.querySelectorAll(".md-code__nav").forEach((nav, index) => {
      if (!(nav instanceof HTMLElement)) return;
      nav.setAttribute("aria-label", `Code block ${index + 1} actions`);
    });
  };

  const initializeModelIdCopyControl = () => {
    const button = document.querySelector("[data-vh-copy-model-id]");
    const label = button?.querySelector("[data-vh-copy-model-id-label]");
    const modelId = button instanceof HTMLButtonElement
      ? button.dataset.modelId?.trim()
      : "";
    if (!(button instanceof HTMLButtonElement)
        || !(label instanceof HTMLElement)
        || !modelId) return;

    const idleLabel = "Copy model ID";
    let resetTimer;
    let copyInProgress = false;

    const copyWithSelection = () => {
      const previousFocus = document.activeElement;
      const copyBuffer = document.createElement("textarea");
      copyBuffer.value = modelId;
      copyBuffer.setAttribute("readonly", "");
      copyBuffer.setAttribute("aria-hidden", "true");
      copyBuffer.style.position = "fixed";
      copyBuffer.style.inset = "0 auto auto -9999px";
      copyBuffer.style.opacity = "0";
      document.body.append(copyBuffer);

      let copied = false;
      try {
        copyBuffer.focus({ preventScroll: true });
        copyBuffer.select();
        copied = document.execCommand("copy");
      } finally {
        copyBuffer.remove();
        if (previousFocus instanceof HTMLElement) previousFocus.focus({ preventScroll: true });
      }
      if (!copied) throw new Error("The browser rejected both clipboard copy methods.");
    };

    const writeModelId = async () => {
      try {
        await navigator.clipboard.writeText(modelId);
      } catch (_error) {
        copyWithSelection();
      }
    };

    const resetFeedback = () => {
      label.textContent = idleLabel;
      button.setAttribute("aria-label", idleLabel);
    };

    label.setAttribute("aria-live", "polite");
    label.setAttribute("aria-atomic", "true");
    resetFeedback();
    button.setAttribute("aria-busy", "false");
    button.addEventListener("click", async () => {
      if (copyInProgress) return;
      copyInProgress = true;
      button.setAttribute("aria-busy", "true");
      window.clearTimeout(resetTimer);
      try {
        await writeModelId();
        label.textContent = "Copied";
        button.setAttribute("aria-label", "Copied");
      } catch (_error) {
        label.textContent = "Copy failed";
        button.setAttribute("aria-label", "Copy failed");
      } finally {
        copyInProgress = false;
        button.setAttribute("aria-busy", "false");
        resetTimer = window.setTimeout(resetFeedback, 1600);
      }
    });
  };

  const initializeModelFactsDisclosure = () => {
    const disclosure = document.querySelector("[data-vh-model-facts-disclosure]");
    if (!(disclosure instanceof HTMLDetailsElement)) return;

    const desktopLayout = window.matchMedia("(min-width: 76.25em)");
    const synchronizeLayout = () => {
      disclosure.open = desktopLayout.matches;
    };
    desktopLayout.addEventListener("change", synchronizeLayout);
    synchronizeLayout();
  };

  const initializeModelSectionNavigation = () => {
    const navigation = document.querySelector(".vh-model-detail__tabs");
    if (!(navigation instanceof HTMLElement)) return;

    const sections = Array.from(navigation.querySelectorAll("a[href^='#']"))
      .flatMap((link) => {
        if (!(link instanceof HTMLAnchorElement) || !link.hash) return [];
        const target = document.getElementById(decodeURIComponent(link.hash.slice(1)));
        return target instanceof HTMLElement ? [{ link, target }] : [];
      })
      .sort((left, right) => left.target.compareDocumentPosition(right.target)
        & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1);
    if (!sections.length) return;
    const defaultLink = navigation.querySelector('[data-vh-model-tab="model-card"]');

    const setCurrent = (currentLink) => {
      sections.forEach(({ link }) => {
        if (link === currentLink) link.setAttribute("aria-current", "location");
        else link.removeAttribute("aria-current");
      });
    };
    const updateFromScroll = () => {
      if (window.scrollY <= 24) {
        setCurrent(defaultLink instanceof HTMLAnchorElement ? defaultLink : sections[0].link);
        return;
      }
      const header = document.querySelector(".md-header");
      const shellThreshold = Math.max(
        0,
        header?.getBoundingClientRect().bottom || 0,
        navigation.getBoundingClientRect().bottom,
      );
      const threshold = Math.max(shellThreshold + 32, window.innerHeight * 0.24);
      let current = defaultLink instanceof HTMLAnchorElement ? defaultLink : sections[0].link;
      sections.forEach(({ link, target }) => {
        if (target.getBoundingClientRect().top <= threshold) current = link;
      });
      setCurrent(current);
    };

    const ANCHOR_RELEASE_DISTANCE = 24;
    let lockedLink = null;
    let settledScrollY = null;
    let settleTimer;

    const settleAnchorNavigation = () => {
      if (!(lockedLink instanceof HTMLAnchorElement) || settledScrollY !== null) return;
      settledScrollY = window.scrollY;
      window.clearTimeout(settleTimer);
    };
    const deferAnchorSettlement = () => {
      window.clearTimeout(settleTimer);
      settleTimer = window.setTimeout(settleAnchorNavigation, 240);
    };
    const releaseAnchorNavigation = () => {
      window.clearTimeout(settleTimer);
      lockedLink = null;
      settledScrollY = null;
    };
    const lockAnchorNavigation = (link) => {
      lockedLink = link;
      settledScrollY = null;
      setCurrent(link);
      deferAnchorSettlement();
    };
    const lockCurrentHash = () => {
      if (!window.location.hash) return false;
      let targetId;
      try {
        targetId = decodeURIComponent(window.location.hash.slice(1));
      } catch (_error) {
        return false;
      }
      const matchingSection = sections.find(({ target }) => target.id === targetId);
      if (!matchingSection) return false;
      lockAnchorNavigation(matchingSection.link);
      return true;
    };

    let updateScheduled = false;
    let releaseOnNextUpdate = false;
    const scheduleUpdate = (event) => {
      if (event?.type === "hashchange") {
        if (lockCurrentHash()) return;
        releaseAnchorNavigation();
      }
      if (event?.type === "scroll" && lockedLink instanceof HTMLAnchorElement) {
        if (settledScrollY === null) deferAnchorSettlement();
        releaseOnNextUpdate = true;
      }
      if (updateScheduled) return;
      updateScheduled = true;
      requestAnimationFrame(() => {
        updateScheduled = false;
        const mayReleaseAnchor = releaseOnNextUpdate;
        releaseOnNextUpdate = false;
        if (lockedLink instanceof HTMLAnchorElement) {
          const movedAfterSettlement = settledScrollY !== null
            && Math.abs(window.scrollY - settledScrollY) >= ANCHOR_RELEASE_DISTANCE;
          if (mayReleaseAnchor && movedAfterSettlement) releaseAnchorNavigation();
          else {
            setCurrent(lockedLink);
            return;
          }
        }
        updateFromScroll();
      });
    };
    navigation.addEventListener("click", (event) => {
      const link = event.target instanceof Element
        ? event.target.closest("a[href^='#']")
        : null;
      if (link instanceof HTMLAnchorElement) lockAnchorNavigation(link);
    });
    window.addEventListener("scroll", scheduleUpdate, { passive: true });
    window.addEventListener("scrollend", settleAnchorNavigation, { passive: true });
    window.addEventListener("resize", scheduleUpdate);
    window.addEventListener("hashchange", scheduleUpdate);
    if (!lockCurrentHash()) scheduleUpdate();
  };

  const initializeSemanticHighlights = () => {
    const optimizationRoute = /\/(?:guides\/[^/]*optimization[^/]*|project\/adding-an-optimization)\/?$/;
    const modelRoute = /\/models\/providers\/[^/]+\/?$/;
    const currentPath = window.location.pathname.replace(/index\.html$/, "");
    const isOptimizationPage = optimizationRoute.test(currentPath);
    const isModelPage = modelRoute.test(currentPath);

    document.body.classList.toggle("vh-optimization-page", isOptimizationPage);
    document.body.classList.toggle("vh-model-page", isModelPage);
    document.querySelectorAll(".md-nav__link[href]").forEach((link) => {
      if (!(link instanceof HTMLAnchorElement)) return;
      let target;
      try {
        target = new URL(link.href, window.location.href).pathname.replace(/index\.html$/, "");
      } catch (error) {
        return;
      }
      link.classList.toggle(
        "vh-model-link",
        modelRoute.test(target) && !target.endsWith("/models/providers/"),
      );
      link.classList.toggle("vh-optimization-link", optimizationRoute.test(target));
    });

    if (!isOptimizationPage) return;
    document.querySelectorAll(".md-typeset code:not(pre code)").forEach((term) => {
      term.classList.add("vh-optimization-term");
    });
  };

  const initializeScrollableRegions = () => {
    const normalizeScrollableRegions = () => {
      document.querySelectorAll(".md-typeset__table").forEach((region, index) => {
        if (!(region instanceof HTMLElement)) return;
        region.tabIndex = 0;
        region.setAttribute("aria-label", `Scrollable table ${index + 1}`);
      });
      document.querySelectorAll(".md-typeset pre > code").forEach((region, index) => {
        if (!(region instanceof HTMLElement)) return;
        const scrollable = region.clientWidth > 0 && region.scrollWidth > region.clientWidth + 1;
        if (scrollable) {
          region.tabIndex = 0;
          region.dataset.vhScrollableCode = "true";
          region.setAttribute("aria-label", `Scrollable code block ${index + 1}`);
        } else if (region.dataset.vhScrollableCode === "true") {
          region.removeAttribute("tabindex");
          region.removeAttribute("aria-label");
          delete region.dataset.vhScrollableCode;
        }
      });
      document.querySelectorAll(".md-typeset .tabbed-labels").forEach((region, index) => {
        if (!(region instanceof HTMLElement)) return;
        const scrollable = region.clientWidth > 0 && region.scrollWidth > region.clientWidth + 1;
        if (scrollable) {
          region.tabIndex = 0;
          region.dataset.vhScrollableTabs = "true";
          region.setAttribute("aria-label", `Scrollable options ${index + 1}`);
        } else if (region.dataset.vhScrollableTabs === "true") {
          region.removeAttribute("tabindex");
          region.removeAttribute("aria-label");
          delete region.dataset.vhScrollableTabs;
        }
      });
      document.querySelectorAll(".md-search-result article pre > code").forEach((region, index) => {
        if (!(region instanceof HTMLElement)) return;
        region.tabIndex = 0;
        region.setAttribute("aria-label", `Search result code ${index + 1}`);
      });
    };

    normalizeScrollableRegions();
    const searchResult = document.querySelector(".md-search-result");
    if (searchResult instanceof HTMLElement) {
      new MutationObserver(normalizeScrollableRegions).observe(searchResult, {
        childList: true,
        subtree: true,
      });
    }
    document.addEventListener("change", (event) => {
      if (!(event.target instanceof HTMLInputElement)
          || !event.target.matches('.tabbed-set > input[type="radio"]')) return;
      requestAnimationFrame(normalizeScrollableRegions);
    });
    window.addEventListener("resize", normalizeScrollableRegions);
  };

  const initializeContentTabFocus = () => {
    let focusRequestId = 0;
    const focusSettleFrames = 6;

    const scrollElementBy = (element, { left = 0, top = 0 }) => {
      if (!(element instanceof HTMLElement)) return;
      const previousScrollBehavior = element.style.scrollBehavior;
      element.style.scrollBehavior = "auto";
      element.scrollLeft += left;
      element.scrollTop += top;
      element.style.scrollBehavior = previousScrollBehavior;
    };

    const scrollViewportBy = (top) => {
      const scrollingElement = document.scrollingElement;
      if (scrollingElement instanceof HTMLElement) {
        scrollElementBy(scrollingElement, { top });
      } else {
        window.scrollBy({ behavior: "instant", top });
      }
    };

    const keepLabelInViewport = (label) => {
      if (!(label instanceof HTMLLabelElement)) return;
      const labels = label.parentElement;
      const viewportMargin = 4;
      let bounds = label.getBoundingClientRect();
      if (labels instanceof HTMLElement) {
        const labelsBounds = labels.getBoundingClientRect();
        const visibleLeft = Math.max(viewportMargin, labelsBounds.left + viewportMargin);
        const visibleRight = Math.min(
          window.innerWidth - viewportMargin,
          labelsBounds.right - viewportMargin,
        );
        if (bounds.left < visibleLeft) {
          scrollElementBy(labels, { left: bounds.left - visibleLeft });
        } else if (bounds.right > visibleRight) {
          scrollElementBy(labels, { left: bounds.right - visibleRight });
        }
      }

      bounds = label.getBoundingClientRect();
      const header = document.querySelector(".md-header");
      const headerBottom = header instanceof HTMLElement
        ? header.getBoundingClientRect().bottom
        : 0;
      const viewportTop = Math.max(viewportMargin, headerBottom + viewportMargin);
      if (bounds.top < viewportTop) {
        scrollViewportBy(bounds.top - viewportTop);
      } else if (bounds.bottom > window.innerHeight - viewportMargin) {
        scrollViewportBy(bounds.bottom - (window.innerHeight - viewportMargin));
      }
    };

    const revealLabel = (input, label, requestId) => {
      if (!(input instanceof HTMLInputElement) || !(label instanceof HTMLLabelElement)) return;
      const isCurrentRequest = () => (
        document.activeElement === input
        && input.dataset.vhFocusRequest === String(requestId)
      );
      const settleLabel = (frame) => {
        if (!isCurrentRequest()) return;
        keepLabelInViewport(label);
        if (frame < focusSettleFrames) {
          requestAnimationFrame(() => settleLabel(frame + 1));
          return;
        }
        input.dataset.vhFocusSettled = String(requestId);
      };
      requestAnimationFrame(() => settleLabel(1));
    };

    document.querySelectorAll(".tabbed-set").forEach((tabSet) => {
      if (!(tabSet instanceof HTMLElement)) return;
      const inputs = Array.from(
        tabSet.querySelectorAll(':scope > input[type="radio"][id]:not(:disabled)'),
      ).filter((input) => input instanceof HTMLInputElement);
      if (!inputs.length) return;

      inputs.forEach((input) => {
        const label = Array.from(
          tabSet.querySelectorAll(".tabbed-labels > label[for]"),
        ).find((candidate) => candidate instanceof HTMLLabelElement && candidate.htmlFor === input.id);
        if (!(label instanceof HTMLLabelElement)) return;

        input.addEventListener("focus", () => {
          const requestId = ++focusRequestId;
          input.dataset.vhFocusRequest = String(requestId);
          delete input.dataset.vhFocusSettled;
          revealLabel(input, label, requestId);
          label.classList.add("vh-content-tab--focus");
        });
        input.addEventListener("blur", () => {
          focusRequestId += 1;
          delete input.dataset.vhFocusRequest;
          delete input.dataset.vhFocusSettled;
          label.classList.remove("vh-content-tab--focus");
        });
      });

      tabSet.addEventListener("keydown", (event) => {
        if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
        if (!(event.target instanceof HTMLInputElement) || !inputs.includes(event.target)) return;
        const direction = event.key === "ArrowRight" || event.key === "ArrowDown"
          ? 1
          : event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1 : 0;
        if (!direction) return;

        event.preventDefault();
        const currentIndex = inputs.indexOf(event.target);
        const nextInput = inputs[(currentIndex + direction + inputs.length) % inputs.length];
        nextInput.checked = true;
        nextInput.dispatchEvent(new Event("input", { bubbles: true }));
        nextInput.dispatchEvent(new Event("change", { bubbles: true }));
        nextInput.focus({ preventScroll: true });
      });
    });
  };

  const initializeSequentialFocusBoundary = () => {
    const skipLink = document.querySelector(".md-skip");
    if (!(skipLink instanceof HTMLAnchorElement)) return;

    document.addEventListener("keydown", (event) => {
      if (event.key !== "Tab" || event.shiftKey || event.altKey || event.ctrlKey || event.metaKey
          || document.activeElement !== document.body) return;
      event.preventDefault();
      skipLink.focus();
    }, true);
  };

  const initializeSourceControl = () => {
    const link = document.querySelector('[data-vh-header-control="source"] a[href]');
    if (!(link instanceof HTMLAnchorElement)) return;

    link.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      event.preventDefault();
      window.location.assign(link.href);
    });
  };

  const initializeHeaderControls = () => {
    initializeShellScrollTracking();
    initializePrimaryNavigationControl();
    initializeTableOfContentsTracking();
    initializeVersionControl();
    initializeSearchControl();
    initializeLanguageControl();
    initializeThemeControl();
    initializeCodeBlockLandmarks();
    initializeModelIdCopyControl();
    initializeModelFactsDisclosure();
    initializeModelSectionNavigation();
    initializeSemanticHighlights();
    initializeScrollableRegions();
    initializeContentTabFocus();
    initializeSequentialFocusBoundary();
    initializeSourceControl();
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initializeHeaderControls, { once: true });
  } else {
    initializeHeaderControls();
  }
})();
