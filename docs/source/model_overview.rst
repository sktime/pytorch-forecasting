.. _model_overview:

Model Overview
==============

This page provides a comprehensive, searchable overview of all forecasting models available in pytorch-forecasting.
The table is automatically generated from the model registry, ensuring it stays up-to-date as new models are added.

Use the search box below to filter models by name, type, or capabilities. Click on column headers to sort the table.

.. raw:: html

   <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
   <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
   <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
   <script src="_static/model_overview.js"></script>
   
   <div id="model-overview-container" style="margin: 20px 0;">
       <div id="model-filters" style="margin-bottom: 15px;">
           <label style="margin-right: 15px;">
               Model Type:
               <select id="type-filter" style="margin-left: 5px; padding: 5px;">
                   <option value="">All</option>
                   <option value="forecaster_pytorch">Forecaster (v1)</option>
                   <option value="forecaster_pytorch_v2">Forecaster (v2)</option>
               </select>
           </label>
           <label>
               Capability:
               <select id="capability-filter" style="margin-left: 5px; padding: 5px;">
                   <option value="">All</option>
                   <option value="Covariates">Covariates</option>
                   <option value="Multiple targets">Multiple targets</option>
                   <option value="Uncertainty">Uncertainty</option>
                   <option value="Flexible history">Flexible history</option>
                   <option value="Cold-start">Cold-start</option>
               </select>
           </label>
       </div>
       <table id="model-table" class="display" style="width:100%"></table>
   </div>

How to Use This Page
--------------------

- **Search**: Type in the search box to filter models by any column
- **Sort**: Click column headers to sort ascending/descending
- **Filter by Type**: Use the "Model Type" dropdown to filter by forecaster version
- **Filter by Capability**: Use the "Capability" dropdown to find models with specific features
- **Click Model Names**: Click on any model name to view its detailed API documentation

.. raw:: html
   :file: _static/model_overview_table.html



The table includes the following information:

- **Model Name**: Name of the model with link to API documentation
- **Type**: Object type (forecaster version)
- **Covariates**: Whether the model supports exogenous variables/covariates
- **Multiple targets**: Whether the model can handle multiple target variables
- **Uncertainty**: Whether the model provides uncertainty estimates
- **Flexible history**: Whether the model supports variable history lengths
- **Cold-start**: Whether the model can make predictions without historical data
- **Compute**: Computational resource requirement (1-5 scale, where 5 is most intensive)

For more information about selecting the right model for your use case, see the :doc:`models` page.

