use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Data, DataStruct, DeriveInput, Expr, Meta, MetaNameValue, Path, parse_macro_input,
    punctuated::Punctuated, token::Comma,
};

#[proc_macro_derive(Model, attributes(model))]
pub fn derive_model(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let struct_name = &input.ident;

    let mut input_type: Option<Path> = None;
    let mut output_type: Option<Path> = None;

    for attr in &input.attrs {
        if attr.path().is_ident("model") {
            if let Meta::List(list) = &attr.meta {
                let nested =
                    list.parse_args_with(Punctuated::<MetaNameValue, Comma>::parse_terminated);
                if let Ok(items) = nested {
                    for item in items {
                        if item.path.is_ident("input") {
                            if let Expr::Path(expr_path) = item.value {
                                input_type = Some(expr_path.path);
                            }
                        } else if item.path.is_ident("output") {
                            if let Expr::Path(expr_path) = item.value {
                                output_type = Some(expr_path.path);
                            }
                        }
                    }
                }
            }
        }
    }

    let input_type = input_type.expect("Missing 'input' parameter in #[model(...)] attribute");
    let output_type = output_type.expect("Missing 'output' parameter in #[model(...)] attribute");

    let fields = match &input.data {
        Data::Struct(DataStruct { fields, .. }) => fields,
        _ => panic!("Model derive macro only supports structs"),
    };

    let filtered_fields: Vec<_> = fields
        .iter()
        .filter(|f| {
            !f.attrs.iter().any(|attr| {
                attr.path().is_ident("model")
                    && if let Meta::List(list) = &attr.meta {
                        list.tokens.to_string() == "ignore"
                    } else {
                        false
                    }
            })
        })
        .collect();
    let field_names: Vec<_> = filtered_fields.iter().map(|f| &f.ident).collect();
    let field_types: Vec<_> = filtered_fields.iter().map(|f| &f.ty).collect();
    let field_idents: Vec<_> = field_names
        .iter()
        .map(|ident| ident.as_ref().map(|i| format_ident!("{}", i)))
        .collect();
    let field_type_checks: Vec<TokenStream> = field_idents
        .iter()
        .zip(field_types.iter())
        .filter_map(|(ident, ty)| {
            ident.as_ref().map(|_i| {
                quote! {
                    const _: () = {
                        fn assert_model<T: nove_model::Model>() {}
                        fn check() { assert_model::<#ty>(); }
                    };
                }
            })
        })
        .collect();

    let require_grad_calls: Vec<TokenStream> = field_idents
        .iter()
        .filter_map(|ident| {
            ident.as_ref().map(|i| {
                quote! {
                    self.#i.require_grad(grad_enabled)?;
                }
            })
        })
        .collect();

    let to_device_calls: Vec<TokenStream> = field_idents
        .iter()
        .filter_map(|ident| {
            ident.as_ref().map(|i| {
                quote! {
                    self.#i.to_device(device)?;
                }
            })
        })
        .collect();

    let to_dtype_calls: Vec<TokenStream> = field_idents
        .iter()
        .filter_map(|ident| {
            ident.as_ref().map(|i| {
                quote! {
                    self.#i.to_dtype(dtype)?;
                }
            })
        })
        .collect();

    let parameters_calls: Vec<TokenStream> = field_idents
        .iter()
        .filter_map(|ident| {
            ident.as_ref().map(|i| {
                quote! {
                    params.extend(self.#i.parameters()?);
                }
            })
        })
        .collect();

    let named_parameters_calls: Vec<TokenStream> = field_idents
        .iter()
        .filter_map(|ident| {
            ident.as_ref().map(|i| {
                quote! {
                    let field_params = self.#i.named_parameters()?;
                    for (name, tensor) in field_params {
                        let prefixed_name = format!("{}.{}", stringify!(#i), name);
                        params.insert(prefixed_name, tensor);
                    }
                }
            })
        })
        .collect();

    let display_fields: Vec<TokenStream> = field_idents
        .iter()
        .filter_map(|ident| {
            ident.as_ref().map(|i| {
                quote! {
                    writeln!(f, "  {}: {}", stringify!(#i), self.#i)?;
                }
            })
        })
        .collect();

    let expanded = quote! {
        #(#field_type_checks)*

        impl nove_model::Model for #struct_name {
            type Input = #input_type;
            type Output = #output_type;

            fn forward(&mut self, input: Self::Input) -> Result<Self::Output, nove_model::ModelError> {
                self.forward(input)
            }

            fn require_grad(&mut self, grad_enabled: bool) -> Result<(), nove_model::ModelError> {
                #(#require_grad_calls)*
                Ok(())
            }

            fn to_device(&mut self, device: &nove_tensor::Device) -> Result<(), nove_model::ModelError> {
                #(#to_device_calls)*
                Ok(())
            }

            fn to_dtype(&mut self, dtype: &nove_tensor::DType) -> Result<(), nove_model::ModelError> {
                #(#to_dtype_calls)*
                Ok(())
            }

            fn parameters(&self) -> Result<Vec<nove_tensor::Tensor>, nove_model::ModelError> {
                let mut params = Vec::new();
                #(#parameters_calls)*
                Ok(params)
            }

            fn named_parameters(&self) -> Result<std::collections::HashMap<String, nove_tensor::Tensor>, nove_model::ModelError> {
                let mut params = std::collections::HashMap::new();
                #(#named_parameters_calls)*
                Ok(params)
            }
        }

        impl std::fmt::Display for #struct_name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                writeln!(f, "{}(", stringify!(#struct_name))?;
                #(#display_fields)*
                write!(f, ")")
            }
        }
    };

    proc_macro::TokenStream::from(expanded)
}
