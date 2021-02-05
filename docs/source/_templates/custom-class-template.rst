{{ fullname.split(".")[-1] | escape | underline}}

.. currentmodule:: {{ module }}


.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :exclude-members: __init__
   {% set allow_inherited = "zero_grad" not in inherited_members %}  {# no inheritance for torch.nn.Modules #}
   {%if allow_inherited %}
   :inherited-members:
   {% endif %}

   {% block methods %}
   {% set allowed_methods = [] %}
   {% for item in methods %}{% if not item.startswith("_") and (item not in inherited_members or allow_inherited) %}
   {% set a=allowed_methods.append(item) %}
   {% endif %}{%- endfor %}
   {% if allowed_methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in allowed_methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% set allowed_attributes = [] %}
   {% for item in attributes %}{% if not item.startswith("_") and (item not in inherited_members or allow_inherited) %}
   {% set a=allowed_attributes.append(item) %}
   {% endif %}{%- endfor %}
   {% if allowed_attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in allowed_attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
