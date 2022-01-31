#include "test.hpp"
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>

struct program_config
{
	std::string config_file_path;
	std::string output_mesh_file;
	std::string mesh_name;
};

program_config init_app(int argc, char* argv[])
{
	google::InitGoogleLogging(argv[0]);

	boost::program_options::options_description cmd_description("command options");

	cmd_description.add_options()
					("help", "help information")
					("f", boost::program_options::value<std::string>(), "path to configure json file - \"/data/../config.json\"")
					("o", boost::program_options::value<std::string>(), "ouput ply mesh file name - \"somemesh.ply\"")
					("m", boost::program_options::value<std::string>(), "ouput mesh name in ply - not required");

	boost::program_options::variables_map vm;
	boost::program_options::store(boost::program_options::parse_command_line(argc, argv, cmd_description), vm);

	if (vm.count("help")) {
		std::cout << cmd_description;
		exit(EXIT_SUCCESS);
	}

	program_config pcfg;

	if (vm.count("f")) // "data/relief/000.json"
		pcfg.config_file_path = vm["f"].as<std::string>();
	
	if (vm.count("o"))
		pcfg.output_mesh_file = vm["o"].as<std::string>();
	
	if (vm.count("m"))
		pcfg.mesh_name = vm["m"].as<std::string>();

	if (pcfg.config_file_path.empty() || pcfg.output_mesh_file.empty())
		throw SfMException("Error: configure file path or output ply name not set!");

	return pcfg;
}


int main(int argc, char* argv[])
{
	program_config pcfg = init_app(argc, argv);

	test_sfm(pcfg.config_file_path, pcfg.output_mesh_file, pcfg.mesh_name);

	
    return 0;
}


