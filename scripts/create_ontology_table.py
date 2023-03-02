import re
from importlib.machinery import SourceFileLoader
from os import listdir
import os
from os.path import join, isdir, isfile
import sys
from enum import EnumMeta, Enum, IntEnum, Flag, IntFlag
import importlib.util
import inspect


def get_directories(directory):
    return [d for d in listdir(directory) if isdir(join(directory, d)) and d != "__pycache__"]


def get_files(directory):
    return [f for f in listdir(directory) if isfile(join(directory, f)) and f != "__init__.py"]


def get_classes(concept_file):
    package = ".".join(concept_file.split("/")[-7:-1]) + "."
    file = os.path.basename(concept_file)

    classes = []
    # Check that the file is a python file but not the init.py
    if not file.endswith('.py') or file == '__init__.py':
        return classes

    # Get the class and module
    class_name = file[:-3]
    class_module = __import__(package + class_name, fromlist=[class_name])

    # Get the frames' class
    module_dict = class_module.__dict__
    for obj in module_dict:
        if isinstance(module_dict[obj], type) and module_dict[obj].__module__ == class_module.__name__:
            classes.append(obj)
    return classes


def get_versions(directory):
    versions = get_directories(directory)
    return sorted([version for version in versions if re.findall(r"^v[0123456789.]+$", version)])


def get_description_from(file_name):
    data = SourceFileLoader(f"init_file_{file_name}", file_name).load_module()
    return data.description if hasattr(data, "description") else ""


def cleanup(entry, level="version"):
    clean = False
    for key in entry.keys():
        if clean is True:
            entry[key] = ""
        if level == key:
            clean = True
    return entry


def write_entry(doc_latex_file, entry):
    values = [value.replace("_", "\_") for value in entry.values()]
    doc_latex_file.write(
f"""
    {" & ".join(values)} \\\\
    \\hline
"""
    )


def process_ontology_versions(versions, ontology_directory, doc_latex_file):
    for i, version in enumerate(versions):
        # Create the ontology entry describing the version, and write it to the documentation table
        entry = {
            "version": version,
            "variable_type": "",
            "namespace": "",
            "domain": "",
            "concept": "",
            "variable": "",
            "unit_or_category": "",
            "description": "",
        }
        version_directory = ontology_directory + "/" + version
        entry["description"] = get_description_from(f"{version_directory}/__init__.py")
        if i != 0:
            write_latex_table_end(doc_latex_file)
            write_latex_table_beginning(doc_latex_file)
        write_entry(doc_latex_file, entry)

        # Process all variables types
        variable_types = get_directories(version_directory)
        process_ontology_variables_types(variable_types, version_directory, doc_latex_file, entry)
        cleanup(entry, level="version")


def process_ontology_variables_types(variable_types, version_directory, doc_latex_file, entry):
    for variable_type in variable_types:
        # Create the variable type description, and write it to the documentation table
        variable_type_directory = version_directory + "/" + variable_type
        entry["variable_type"] = variable_type
        entry["description"] = get_description_from(f"{variable_type_directory}/__init__.py")
        write_entry(doc_latex_file, entry)

        # Process all namespaces
        namespaces = get_directories(variable_type_directory)
        process_ontology_namespaces(namespaces, variable_type_directory, doc_latex_file, entry)
        cleanup(entry, level="variable_type")


def process_ontology_namespaces(namespaces, variable_type_directory, doc_latex_file, entry):
    for namespace in namespaces:
        # Create the namespace description, and write it to the documentation table
        namespace_directory = variable_type_directory + "/" + namespace
        entry["namespace"] = namespace
        entry["description"] = get_description_from(f"{namespace_directory}/__init__.py")
        write_entry(doc_latex_file, entry)

        # Process all domains
        domains = get_directories(namespace_directory)
        process_ontology_domains(domains, namespace_directory, doc_latex_file, entry)
        cleanup(entry, level="namespace")


def process_ontology_domains(domains, namespace_directory, doc_latex_file, entry):
    for domain in domains:
        # Create the domain description, and write it to the documentation table
        domain_directory = namespace_directory + "/" + domain
        entry["domain"] = domain
        entry["description"] = get_description_from(f"{domain_directory}/__init__.py")
        write_entry(doc_latex_file, entry)

        # Process all concepts
        concepts = get_files(domain_directory)
        process_ontology_concepts(concepts, domain_directory, doc_latex_file, entry)
        cleanup(entry, level="domain")


def process_ontology_concepts(concepts, domain_directory, doc_latex_file, entry):
    for concept in concepts:
        # Create the concept description, and write it to the documentation table
        concept_file = domain_directory + "/" + concept
        entry["concept"] = concept.replace(".py", "")
        entry["description"] = get_description_from(f"{domain_directory}/{concept}")
        write_entry(doc_latex_file, entry)

        # Process all variables
        variables = get_classes(concept_file)
        process_ontology_variables(variables, concept_file, doc_latex_file, entry)
        cleanup(entry, level="concept")


def get_enums_description(file_name, entry):
    # Retrieve all enumerations in the file
    module_name = \
        f"open_science_network.ontology.{entry['version']}.{entry['variable_type']}.{entry['namespace']}.{entry['domain']}"
    spec = importlib.util.spec_from_file_location(module_name, file_name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    variables = {
        item: getattr(mod, item) for item in dir(mod) if not item.startswith("__")
    }
    enums = {
        name: value
        for name, value in variables.items()
        if inspect.isclass(value) and issubclass(
            value, (EnumMeta, Enum, IntEnum, Flag, IntFlag)
        ) and value not in (EnumMeta, Enum, IntEnum, Flag, IntFlag)
    }

    # Retrieve docstring from all enumerations in the file
    docstrings = {name: value.__doc__ for name, value in enums.items()}
    return enums, docstrings


def process_ontology_variables(variables, concept_file, doc_latex_file, entry):
    enums, descriptions = get_enums_description(concept_file, entry)
    for variable in variables:
        # Create the domain description, and write it to the documentation table
        entry["variable"] = variable
        entry["unit_or_category"] = process_unit_or_category(enums[variable].__members__.keys())
        entry["description"] = descriptions[variable]
        write_entry(doc_latex_file, entry)


def process_unit_or_category(enum_values):
    enum_values = list(enum_values)
    if set(enum_values) == {"Yes", "No"}:
        return "Boolean"
    if set(enum_values) == {"Continuous"}:
        return "Continuous"
    return f"Categorical"


def create_ontology_table(ontology_directory, doc_latex_file):
    """
    Create the ontology documentation table from the file system
    :param ontology_directory: the directory containing the ontology
    :param doc_latex_file: the latex file in which to write the ontology table
    """
    versions = get_versions(ontology_directory)
    process_ontology_versions(versions, ontology_directory, doc_latex_file)


def write_latex_file_beginning(latex_file):
    """
    Write the lines at the beginning of the latex file that are required to compile the file to PDF format
    :param latex_file: the latex file
    """
    latex_file.write(
"""
\\documentclass[border=1in]{standalone}
\\usepackage{array}

\\begin{document}
"""
    )
    write_latex_table_beginning(latex_file)


def write_latex_table_beginning(latex_file):
    latex_file.write(
"""
\\bgroup
\\def\\arraystretch{1.5}
\\begin{tabular}{ | m{1.5cm} | m{2.5cm} | m{2.5cm} | m{2.5cm} | m{2.5cm} | m{4cm} | m{2.5cm}| m{5cm} | }
    \\multicolumn{1}{c}{\\textbf{Version}} & \\multicolumn{1}{c}{\\textbf{Variable type}} & \\multicolumn{1}{c}{\\textbf{Namespace}} & \\multicolumn{1}{c}{\\textbf{Domain}} & \\multicolumn{1}{c}{\\textbf{Concept}} & \\multicolumn{1}{c}{\\textbf{Variable}} & \\multicolumn{1}{c}{\\textbf{Unit or Category}} & \\multicolumn{1}{c}{\\textbf{Description}} \\\\
    \\hline
"""
)


def write_latex_table_end(latex_file):
    latex_file.write(
"""
\\end{tabular}
\\egroup
"""
    )


def write_latex_file_end(latex_file):
    """
    Write the lines at the end of the latex file that are required to compile the file to PDF format
    :param latex_file: the latex file
    """
    write_latex_table_end(latex_file)
    latex_file.write(
"""
\\end{document}
"""
    )


if __name__ == '__main__':
    # Create ontology documentation folder
    ontology_doc_dir = sys.argv[1] + "/documentation/"
    if not os.path.isdir(ontology_doc_dir):
        os.makedirs(ontology_doc_dir)

    # Create latex file containing the ontology documentation table
    with open(ontology_doc_dir + sys.argv[2] + ".tex", "w+") as latex_file:
        write_latex_file_beginning(latex_file)
        create_ontology_table(sys.argv[1], latex_file)
        write_latex_file_end(latex_file)

    # Compile latex file into a PDF file
    os.system(f"cd {ontology_doc_dir} && pdflatex {sys.argv[2]}.tex")
