#include "App_Module.h"


int main(int argc, char** argv)
{
    bool validation = false;

    std::string validate_data ="";

    if(argc == 2)
    {
        validate_data = argv[1];

        if(validate_data == "false")
        {
            validation = false;
        }

        else if(validate_data == "true")
        {
            validation = true;
        }
        else
        {
            throw std::runtime_error("Unknow bool type ");
        }
       
    }
    else if (argc == 1 && (validate_data ==""))
    {
        validation = false;
        
    }


    App_Module app(validation);

    app.compute_n_visualise();
    
    return 0;

}












