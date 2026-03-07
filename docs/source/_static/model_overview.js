/**
 * JavaScript for interactive model overview table.
 * 
 * This script loads the model overview data from JSON and creates
 * an interactive DataTable with search and filtering capabilities.
 */

$(document).ready(function() {
    // Determine the correct path to the JSON file
    // In built HTML, the path should be relative to the current page
    var jsonPath = '_static/model_overview_db.json';
    
    // Load model data from JSON
    $.getJSON(jsonPath, function(data) {
        // Initialize DataTable
        var table = $('#model-table').DataTable({
            data: data,
            columns: [
                { 
                    data: 'Model Name', 
                    title: 'Model Name',
                    render: function(data, type, row) {
                        // If data is already HTML (from pandas), return as-is
                        if (type === 'display' && data && data.includes('<a')) {
                            return data;
                        }
                        return data || row['Model Name'] || '';
                    }
                },
                { data: 'Type', title: 'Type' },
                { data: 'Covariates', title: 'Covariates' },
                { data: 'Multiple targets', title: 'Multiple targets' },
                { data: 'Uncertainty', title: 'Uncertainty' },
                { data: 'Flexible history', title: 'Flexible history' },
                { data: 'Cold-start', title: 'Cold-start' },
                { data: 'Compute', title: 'Compute' }
            ],
            pageLength: 25,
            order: [[0, 'asc']],
            responsive: true,
            dom: 'lfrtip',
            language: {
                search: "Search models:",
                lengthMenu: "Show _MENU_ models per page",
                info: "Showing _START_ to _END_ of _TOTAL_ models",
                infoEmpty: "No models found",
                infoFiltered: "(filtered from _MAX_ total models)"
            }
        });
        
        // Filter by type
        $('#type-filter').on('change', function() {
            var val = $(this).val();
            table.column(1).search(val).draw();
        });
        
        // Filter by capability
        $('#capability-filter').on('change', function() {
            var val = $(this).val();
            if (val === '') {
                // Clear all capability filters
                table.columns([2, 3, 4, 5, 6]).search('').draw();
            } else {
                // Map capability name to column index
                var capabilityMap = {
                    'Covariates': 2,
                    'Multiple targets': 3,
                    'Uncertainty': 4,
                    'Flexible history': 5,
                    'Cold-start': 6
                };
                
                var colIdx = capabilityMap[val];
                if (colIdx !== undefined) {
                    // Clear all capability columns first
                    table.columns([2, 3, 4, 5, 6]).search('');
                    // Then search in the specific column
                    table.column(colIdx).search('✓').draw();
                } else {
                    // If capability not found, search in all columns
                    table.search(val).draw();
                }
            }
        });
        
        // Clear filters when "All" is selected
        $('#type-filter, #capability-filter').on('change', function() {
            if ($(this).val() === '') {
                if ($(this).attr('id') === 'type-filter') {
                    table.column(1).search('').draw();
                }
            }
        });
    }).fail(function(jqXHR, textStatus, errorThrown) {
        // Handle error loading JSON
        console.error('Error loading model overview data:', textStatus, errorThrown);
        $('#model-table').html(
            '<tr><td colspan="8" style="text-align: center; padding: 20px;">' +
            'Error loading model overview data. Please ensure the documentation was built correctly.' +
            '</td></tr>'
        );
    });
});

