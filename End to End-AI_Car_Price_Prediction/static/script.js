document.addEventListener("DOMContentLoaded", () => {

    const mileage = document.getElementById("mileage");
    const engine = document.getElementById("engine");
    const maxPower = document.getElementById("max_power");
    const seats = document.getElementById("seats");

    const brandElement = document.getElementById("brand");
    const modelElement = document.getElementById("model");

    const brandSelect = new TomSelect("#brand", {
        create: false,
        maxItems: 1,
        closeAfterSelect: true,
        searchField: "text",
        sortField: {
            field: "text",
            direction: "asc"
        },
        placeholder: "Search Brand..."
    });

    const modelSelect = new TomSelect("#model", {
        create: false,
        maxItems: 1,
        closeAfterSelect: true,
        searchField: "text",
        sortField: {
            field: "text",
            direction: "asc"
        },
        placeholder: "Search Model..."
    });

    brandSelect.on("item_add", () => {
        brandSelect.close();
        brandSelect.blur();
    });

    modelSelect.on("item_add", () => {
        modelSelect.close();
        modelSelect.blur();
    });

    modelSelect.disable();

    function clearSpecs() {
        mileage.value = "";
        engine.value = "";
        maxPower.value = "";
        seats.value = "";
    }

    async function loadModels(brand, selectedModel = "") {

        modelSelect.clearOptions();

        if (!selectedModel) {
            modelSelect.clear(true);
            clearSpecs();
        }

        modelSelect.disable();

        if (!brand) return;

        try {

            const response = await fetch(
                `/get_models/${encodeURIComponent(brand)}`
            );

            if (!response.ok)
                throw new Error("Unable to fetch models");

            const models = await response.json();

            models.forEach(model => {

                modelSelect.addOption({
                    value: model,
                    text: model
                });

            });

            modelSelect.refreshOptions(false);
            modelSelect.enable();

            if (selectedModel && models.includes(selectedModel)) {

                setTimeout(() => {

                    modelSelect.setValue(selectedModel);

                    modelSelect.close();
                    modelSelect.blur();

                    loadSpecs(brand, selectedModel);

                }, 50);

            }

        }

        catch (err) {

            console.error(err);
            alert("Unable to load models.");

        }

    }

    async function loadSpecs(brand, model) {

        if (!brand || !model) {
            clearSpecs();
            return;
        }

        try {

            const response = await fetch(
                `/get_specs/${encodeURIComponent(brand)}/${encodeURIComponent(model)}`
            );

            if (!response.ok)
                throw new Error("Unable to fetch specs");

            const specs = await response.json();

            mileage.value = specs.mileage ?? "";
            engine.value = specs.engine ?? "";
            maxPower.value = specs.max_power ?? "";
            seats.value = specs.seats ?? "";

        }

        catch (err) {

            console.error(err);
            clearSpecs();

        }

    }

    brandSelect.on("change", (brand) => {

        loadModels(brand);

    });

    modelSelect.on("change", (model) => {

        loadSpecs(
            brandSelect.getValue(),
            model
        );

    });

    // ---------- Restore values after prediction ----------

    const savedBrand = brandElement.dataset.selected;
    const savedModel = modelElement.dataset.selected;

    if (savedBrand) {

        brandSelect.setValue(savedBrand);

        loadModels(savedBrand, savedModel);

    }

});