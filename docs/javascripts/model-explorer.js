(() => {
  const normalize = (value) => String(value || "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();

  const initializeModelExplorer = () => {
    const explorer = document.querySelector("[data-vh-model-explorer]");
    if (!(explorer instanceof HTMLElement)) return;

    const form = explorer.querySelector("[data-vh-model-filters]");
    const query = explorer.querySelector("[data-vh-model-query]");
    const selects = Array.from(explorer.querySelectorAll("[data-vh-model-select]"));
    const checkboxes = Array.from(explorer.querySelectorAll("[data-vh-model-checkbox]"));
    const sort = explorer.querySelector("[data-vh-model-sort]");
    const results = explorer.querySelector("[data-vh-model-results]");
    const cards = Array.from(explorer.querySelectorAll("[data-vh-model-card]"));
    const resultCount = explorer.querySelector("[data-vh-model-result-count]");
    const resultLabel = explorer.querySelector("[data-vh-model-result-label]");
    const activeFilters = explorer.querySelector("[data-vh-model-active-filters]");
    const emptyState = explorer.querySelector("[data-vh-model-empty]");
    const clearButtons = Array.from(explorer.querySelectorAll("[data-vh-model-clear]"));
    if (!(form instanceof HTMLFormElement)
        || !(query instanceof HTMLInputElement)
        || !(sort instanceof HTMLSelectElement)
        || !(results instanceof HTMLElement)
        || !(resultCount instanceof HTMLElement)
        || !(resultLabel instanceof HTMLElement)
        || !(activeFilters instanceof HTMLElement)
        || !(emptyState instanceof HTMLElement)
        || selects.some((select) => !(select instanceof HTMLSelectElement))
        || checkboxes.some((checkbox) => !(checkbox instanceof HTMLInputElement))) return;

    const languageSelect = selects.find((select) => select.name === "language");
    const languageNames = new Map();
    if (languageSelect instanceof HTMLSelectElement) {
      const languageCounts = new Map();
      cards.forEach((card) => {
        new Set((card.dataset.languages || "").split(" ").filter(Boolean)).forEach((code) => {
          languageCounts.set(code, (languageCounts.get(code) || 0) + 1);
        });
      });
      let displayNames;
      if (typeof Intl.DisplayNames === "function") {
        try {
          displayNames = new Intl.DisplayNames(
            [document.documentElement.lang || "en"],
            { type: "language" },
          );
        } catch (_error) {
          displayNames = undefined;
        }
      }
      const languageOptions = Array.from(languageCounts, ([code, count]) => {
        let displayName;
        try {
          displayName = displayNames?.of(code.replace("_", "-"));
        } catch (_error) {
          displayName = undefined;
        }
        if (displayName && normalize(displayName) !== normalize(code)) {
          languageNames.set(code.toLowerCase(), displayName);
        }
        return { code, count, displayName: displayName || code };
      }).sort((left, right) => (
        left.displayName.localeCompare(right.displayName)
        || left.code.localeCompare(right.code)
      ));
      const options = document.createDocumentFragment();
      languageOptions.forEach(({ code, count, displayName }) => {
        const option = document.createElement("option");
        option.value = code;
        option.textContent = displayName === code
          ? `${code} (${count})`
          : `${displayName} · ${code} (${count})`;
        options.append(option);
      });
      languageSelect.append(options);
    }

    const models = cards.map((card) => {
      const languages = (card.dataset.languages || "").split(" ").filter(Boolean);
      const capabilities = new Set(
        (card.dataset.capabilities || "").split(" ").filter(Boolean),
      );
      const languageSearch = languages
        .map((code) => languageNames.get(code.toLowerCase()) || "")
        .join(" ");
      return {
        card,
        name: card.dataset.name || "",
        task: card.dataset.task || "",
        training: card.dataset.training || "",
        trainingRank: Number(card.dataset.trainingRank || 0),
        checkpoint: card.dataset.checkpoint || "",
        license: card.dataset.license || "",
        architecture: card.dataset.architecture || "",
        languageKind: card.dataset.languageKind || "",
        languageCount: Number(card.dataset.languageCount || 0),
        languages,
        capabilities,
        resources: new Set((card.dataset.resources || "").split(" ").filter(Boolean)),
        search: normalize(
          `${card.dataset.search || ""} ${languages.join(" ")} `
          + `${Array.from(capabilities).join(" ")} ${languageSearch}`,
        ),
      };
    });

    const stateKeys = {
      query: "model_q",
      language: "model_language",
      task: "model_task",
      training: "model_training",
      checkpoint: "model_checkpoint",
      license: "model_license",
      architecture: "model_architecture",
      feature: "model_features",
      resource: "model_resources",
      sort: "model_sort",
    };

    const restoreState = () => {
      const parameters = new URL(window.location.href).searchParams;
      query.value = parameters.get(stateKeys.query) || "";
      selects.forEach((select) => {
        const value = parameters.get(stateKeys[select.name]);
        if (value && Array.from(select.options).some((option) => option.value === value)) {
          select.value = value;
        }
      });
      checkboxes.forEach((checkbox) => {
        const selected = new Set(
          (parameters.get(stateKeys[checkbox.name]) || "").split(",").filter(Boolean),
        );
        checkbox.checked = selected.has(checkbox.value);
      });
      const sortValue = parameters.get(stateKeys.sort);
      if (sortValue && Array.from(sort.options).some((option) => option.value === sortValue)) {
        sort.value = sortValue;
      }
    };

    const selectedCheckboxValues = (name) => checkboxes
      .filter((checkbox) => checkbox.name === name && checkbox.checked)
      .map((checkbox) => checkbox.value);

    const currentState = () => ({
      query: query.value.trim(),
      selects: Object.fromEntries(selects.map((select) => [select.name, select.value])),
      features: selectedCheckboxValues("feature"),
      resources: selectedCheckboxValues("resource"),
      sort: sort.value,
    });

    const modelSupportsLanguage = (model, requestedLanguage) => {
      if (!requestedLanguage) return true;
      if (requestedLanguage === "not-text-conditioned") {
        return model.languageKind === "not-text-conditioned";
      }
      const requested = requestedLanguage.toLowerCase();
      return model.languages.some((code) => {
        const candidate = code.toLowerCase();
        return candidate === requested
          || candidate.startsWith(`${requested}-`)
          || requested.startsWith(`${candidate}-`);
      });
    };

    const modelMatches = (model, state) => {
      const queryTokens = normalize(state.query).split(" ").filter(Boolean);
      if (!queryTokens.every((token) => model.search.includes(token))) return false;
      if (!modelSupportsLanguage(model, state.selects.language)) return false;
      for (const key of ["task", "training", "checkpoint", "license", "architecture"]) {
        if (state.selects[key] && model[key] !== state.selects[key]) return false;
      }
      if (!state.features.every((feature) => model.capabilities.has(feature))) return false;
      if (!state.resources.every((resource) => model.resources.has(resource))) return false;
      return true;
    };

    const compareModels = (left, right, order) => {
      const byName = left.name.localeCompare(right.name);
      if (order === "languages") return right.languageCount - left.languageCount || byName;
      if (order === "task") return left.task.localeCompare(right.task) || byName;
      if (order === "training") return left.trainingRank - right.trainingRank || byName;
      return byName;
    };

    const updateUrl = (state) => {
      try {
        const url = new URL(window.location.href);
        Object.values(stateKeys).forEach((key) => url.searchParams.delete(key));
        if (state.query) url.searchParams.set(stateKeys.query, state.query);
        Object.entries(state.selects).forEach(([name, value]) => {
          if (value) url.searchParams.set(stateKeys[name], value);
        });
        if (state.features.length) {
          url.searchParams.set(stateKeys.feature, state.features.join(","));
        }
        if (state.resources.length) {
          url.searchParams.set(stateKeys.resource, state.resources.join(","));
        }
        if (state.sort !== "name") url.searchParams.set(stateKeys.sort, state.sort);
        window.history.replaceState(window.history.state, "", url);
      } catch (_error) {
        // Filtering remains functional in embedded contexts without History access.
      }
    };

    const filterLabel = (control) => {
      if (control instanceof HTMLSelectElement) {
        const fieldLabel = form.querySelector(`label[for="${control.id}"]`)?.textContent?.trim();
        const valueLabel = control.selectedOptions[0]?.textContent?.replace(/\s+\(\d+\)$/, "").trim();
        return `${fieldLabel}: ${valueLabel}`;
      }
      return control.dataset.filterLabel || control.value;
    };

    const addActiveFilterButton = (label, remove) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "vh-model-active-filter";
      button.append(document.createTextNode(label));
      const icon = document.createElement("span");
      icon.setAttribute("aria-hidden", "true");
      icon.textContent = "×";
      button.append(icon);
      button.setAttribute("aria-label", `Remove filter: ${label}`);
      button.addEventListener("click", () => {
        remove();
        applyFilters();
      });
      activeFilters.append(button);
    };

    const renderActiveFilters = (state) => {
      activeFilters.replaceChildren();
      if (state.query) {
        addActiveFilterButton(`Search: “${state.query}”`, () => { query.value = ""; });
      }
      selects.filter((select) => select.value).forEach((select) => {
        addActiveFilterButton(filterLabel(select), () => { select.value = ""; });
      });
      checkboxes.filter((checkbox) => checkbox.checked).forEach((checkbox) => {
        addActiveFilterButton(filterLabel(checkbox), () => { checkbox.checked = false; });
      });
      activeFilters.hidden = activeFilters.childElementCount === 0;
    };

    const applyFilters = () => {
      const state = currentState();
      const orderedModels = [...models].sort((left, right) => compareModels(left, right, state.sort));
      let visibleCount = 0;
      orderedModels.forEach((model) => {
        const matches = modelMatches(model, state);
        model.card.hidden = !matches;
        if (matches) visibleCount += 1;
        results.append(model.card);
      });
      resultCount.textContent = String(visibleCount);
      resultLabel.textContent = visibleCount === 1 ? "model" : "models";
      emptyState.hidden = visibleCount !== 0;
      const hasFilters = Boolean(
        state.query
        || Object.values(state.selects).some(Boolean)
        || state.features.length
        || state.resources.length,
      );
      clearButtons.forEach((button) => { button.hidden = !hasFilters; });
      renderActiveFilters(state);
      updateUrl(state);
    };

    const clearFilters = () => {
      form.reset();
      sort.value = "name";
      applyFilters();
      query.focus({ preventScroll: true });
    };

    form.addEventListener("submit", (event) => event.preventDefault());
    form.addEventListener("input", applyFilters);
    form.addEventListener("change", applyFilters);
    sort.addEventListener("change", applyFilters);
    clearButtons.forEach((button) => button.addEventListener("click", clearFilters));
    query.addEventListener("keydown", (event) => {
      if (event.key !== "Escape" || !query.value) return;
      event.preventDefault();
      query.value = "";
      applyFilters();
    });
    document.addEventListener("keydown", (event) => {
      const target = event.target;
      const acceptsText = target instanceof HTMLInputElement
        || target instanceof HTMLTextAreaElement
        || target instanceof HTMLSelectElement
        || (target instanceof HTMLElement && target.isContentEditable);
      if (event.key !== "/" || acceptsText || event.altKey || event.ctrlKey || event.metaKey) return;
      event.preventDefault();
      query.focus();
    });

    restoreState();
    explorer.dataset.enhanced = "true";
    applyFilters();
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initializeModelExplorer, { once: true });
  } else {
    initializeModelExplorer();
  }
})();
